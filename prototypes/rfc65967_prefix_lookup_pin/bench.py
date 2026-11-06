# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: stock 3-request peer reuse vs 1-request lookup-and-pin.

CPU-only mock. There is no GPU, no NIXL transfer, and no model load.
``a_compute_tokens`` is the exact prefill the executed flow schedules.
``b_ttft_control_us`` is a measured localhost TCP round trip of that
control plane, not a GPU time-to-first-token.

Run from the repo root:

    python3 -m prototypes.rfc65967_prefix_lookup_pin.bench
"""

from __future__ import annotations

import json
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

from prototypes.rfc65967_prefix_lookup_pin.flows import (
    Cluster,
    make_cluster,
    prepare,
    run_proposed,
    run_stock,
)
from prototypes.rfc65967_prefix_lookup_pin.protocol import (
    KvIdentity,
    PinReply,
    block_hashes,
)

BLOCK_SIZE = 16
LENGTHS = (256, 1024, 4096, 16384)
SCENARIOS = ("stale", "hit", "partial")
ITERS = 21
LOOKUP_ITERS = 100
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _p50(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _rpc(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload).encode()
    last_error: OSError | None = None
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(encoded)
                sock.shutdown(socket.SHUT_WR)
                chunks: list[bytes] = []
                while True:
                    piece = sock.recv(1 << 16)
                    if not piece:
                        break
                    chunks.append(piece)
            return json.loads(b"".join(chunks))
        except ConnectionRefusedError as exc:
            last_error = exc
            time.sleep(0.01)
    assert last_error is not None
    raise last_error


class RpcServer:
    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]]):
        self._handler = handler
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(128)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            chunks: list[bytes] = []
            while True:
                piece = conn.recv(1 << 16)
                if not piece:
                    break
                chunks.append(piece)
            if not chunks:
                return
            response = self._handler(json.loads(b"".join(chunks)))
            conn.sendall(json.dumps(response).encode())


def _tokens(n: int) -> list[int]:
    return list(range(n))


def _measure_lookup_us(cluster: Cluster, scenario: str, n_tokens: int) -> float:
    """p50 of lookup-and-pin on the scenario's cache. No hashing, no TCP."""

    tokens = _tokens(n_tokens)
    prepare(cluster, scenario, tokens)
    hashes = block_hashes(tokens, cluster.identity)
    samples: list[float] = []
    for _ in range(LOOKUP_ITERS):
        started = time.perf_counter_ns()
        reply = cluster.A.lookup_raw(hashes, cluster.identity)
        samples.append((time.perf_counter_ns() - started) / 1000)
        if reply.pin_id and not cluster.A.index.complete_pull(reply.pin_id):
            raise RuntimeError("lookup pin was not held")
    return _p50(samples)


def _stock_client(a_port: int, b_port: int, tokens: list[int]) -> None:
    probed = _rpc(a_port, {"op": "stock_probe", "tokens": tokens})
    _rpc(
        b_port,
        {"op": "stock_warmup", "tokens": tokens, "reply": probed["reply"]},
    )
    _rpc(b_port, {"op": "local_generate", "tokens": tokens})


def _proposed_client(b_port: int, tokens: list[int], identity: KvIdentity) -> None:
    _rpc(
        b_port,
        {
            "op": "peer_generate",
            "tokens": tokens,
            "identity": identity.to_json(),
        },
    )


def _build_servers(cluster: Cluster) -> tuple[RpcServer, RpcServer]:
    def handle_a(msg: dict[str, Any]) -> dict[str, Any]:
        op = msg["op"]
        if op == "stock_probe":
            reply = cluster.A.stock_probe(msg["tokens"])
            return {"reply": reply.to_json()}
        if op == "lookup":
            identity = KvIdentity.from_json(msg["identity"])
            hashes = tuple(bytes.fromhex(item) for item in msg["hashes"])
            reply = cluster.A.lookup_raw(hashes, identity)
            return {"reply": reply.to_json()}
        raise ValueError(op)

    def handle_b(msg: dict[str, Any]) -> dict[str, Any]:
        op = msg["op"]
        tokens = msg.get("tokens")
        if op == "stock_warmup":
            reply = PinReply.from_json(msg["reply"])
            intact = cluster.B.stock_warmup(tokens, reply, cluster.A)
            return {"intact": intact}
        if op == "local_generate":
            return {"compute": cluster.B.local_generate(tokens)}
        if op == "peer_generate":
            identity = KvIdentity.from_json(msg["identity"])
            hashes = block_hashes(tokens, cluster.identity)
            remote = _rpc(
                a_server.port,
                {
                    "op": "lookup",
                    "hashes": [item.hex() for item in hashes],
                    "identity": identity.to_json(),
                },
            )
            reply = PinReply.from_json(remote["reply"])
            status = cluster.B.pull_pinned(tokens, reply, cluster.A)
            compute = cluster.B.local_generate(tokens)
            return {
                "status": status,
                "compute": compute,
                "hit_tokens": reply.hit_tokens,
                "miss_reason": reply.miss_reason.value,
            }
        raise ValueError(op)

    a_server = RpcServer(handle_a)
    b_server = RpcServer(handle_b)
    cluster.A.set_endpoint("127.0.0.1", a_server.port)
    cluster.B.set_endpoint("127.0.0.1", b_server.port)
    return a_server, b_server


def _timed(fn: Callable[[], None]) -> float:
    started = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - started) / 1000


def run() -> dict[str, Any]:
    cluster = make_cluster(num_blocks=16384, lease_s=30.0)
    a_server, b_server = _build_servers(cluster)
    rows: list[dict[str, Any]] = []
    try:
        for scenario in SCENARIOS:
            for n_tokens in LENGTHS:
                tokens = _tokens(n_tokens)
                lookup_us = _measure_lookup_us(cluster, scenario, n_tokens)
                prepare(cluster, scenario, tokens)
                in_stock = run_stock(cluster, scenario, tokens)
                prepare(cluster, scenario, tokens)
                in_proposed = run_proposed(cluster, scenario, tokens)

                stock_us: list[float] = []
                proposed_us: list[float] = []
                for _ in range(ITERS):
                    prepare(cluster, scenario, tokens)
                    stock_us.append(
                        _timed(
                            lambda tokens=tokens: _stock_client(
                                a_server.port, b_server.port, tokens
                            )
                        )
                    )
                    a_after = cluster.A.compute_tokens
                    b_after = cluster.B.compute_tokens
                    if a_after != in_stock.a_compute_tokens:
                        raise RuntimeError(
                            "socket stock A compute "
                            f"{a_after} != {in_stock.a_compute_tokens}"
                        )
                    if b_after != in_stock.b_compute_tokens:
                        raise RuntimeError(
                            "socket stock B compute "
                            f"{b_after} != {in_stock.b_compute_tokens}"
                        )
                    prepare(cluster, scenario, tokens)
                    proposed_us.append(
                        _timed(
                            lambda: _proposed_client(
                                b_server.port, tokens, cluster.identity
                            )
                        )
                    )
                    if cluster.A.compute_tokens != in_proposed.a_compute_tokens:
                        raise RuntimeError("socket proposed changed A's compute")
                    if cluster.B.compute_tokens != in_proposed.b_compute_tokens:
                        raise RuntimeError("socket proposed changed B's compute")

                rows.append(
                    {
                        "scenario": scenario,
                        "prompt_tokens": n_tokens,
                        "stock_a_compute_tokens": in_stock.a_compute_tokens,
                        "proposed_a_compute_tokens": in_proposed.a_compute_tokens,
                        "stock_b_compute_tokens": in_stock.b_compute_tokens,
                        "proposed_b_compute_tokens": in_proposed.b_compute_tokens,
                        "stock_critical_path_tokens": (
                            in_stock.a_compute_tokens + in_stock.b_compute_tokens
                        ),
                        "proposed_critical_path_tokens": in_proposed.b_compute_tokens,
                        "stock_pulled_tokens": in_stock.pulled_tokens,
                        "proposed_pulled_tokens": in_proposed.pulled_tokens,
                        "stock_http_requests": in_stock.http_requests,
                        "proposed_http_requests": in_proposed.http_requests,
                        "proposed_sidechannel_rpcs": in_proposed.sidechannel_rpcs,
                        "proposed_miss_reason": in_proposed.miss_reason,
                        "proposed_hit_kind": in_proposed.hit_kind,
                        "stock_b_ttft_control_us_p50": _p50(stock_us),
                        "proposed_b_ttft_control_us_p50": _p50(proposed_us),
                        "lookup_and_pin_us_p50": lookup_us,
                    }
                )

        prepare(cluster, "lease", _tokens(1024))
        lease_stock = run_stock(cluster, "lease", _tokens(1024))
        prepare(cluster, "lease", _tokens(1024))
        lease_proposed = run_proposed(cluster, "lease", _tokens(1024))
        prepare(cluster, "identity", _tokens(1024))
        identity = run_proposed(cluster, "identity", _tokens(1024))
    finally:
        a_server.close()
        b_server.close()

    _check(rows, lease_stock, lease_proposed, identity)
    return {
        "label": "CPU-only mock",
        "gpu": False,
        "nixl_transfer": False,
        "model_weights": False,
        "block_size": BLOCK_SIZE,
        "iterations": ITERS,
        "lookup_iterations": LOOKUP_ITERS,
        "notes": [
            "A's compute tokens are the prefill scheduled by the executed flow.",
            "lookup_and_pin_us_p50 is the in-process call for that scenario.",
            "It excludes block hashing and TCP.",
            "B TTFT is localhost TCP of the JSON body, not GPU TTFT.",
            "No us-per-token constant was applied.",
            "Stock full hit still recomputes one block on A (logits + alignment).",
            "Proposed stale miss moves that prefill onto B and leaves A at 0.",
        ],
        "rows": rows,
        "lease_1024": {
            "stock_stale_reads": lease_stock.stale_reads,
            "stock_lease_refuses": lease_stock.lease_refuses,
            "stock_a_compute_tokens": lease_stock.a_compute_tokens,
            "stock_pulled_tokens": lease_stock.pulled_tokens,
            "proposed_stale_reads": lease_proposed.stale_reads,
            "proposed_lease_refuses": lease_proposed.lease_refuses,
            "proposed_a_compute_tokens": lease_proposed.a_compute_tokens,
            "proposed_b_compute_tokens": lease_proposed.b_compute_tokens,
            "proposed_pulled_tokens": lease_proposed.pulled_tokens,
        },
        "identity_1024": {
            "miss_reason": identity.miss_reason,
            "a_compute_tokens": identity.a_compute_tokens,
            "pulled_tokens": identity.pulled_tokens,
            "b_compute_tokens": identity.b_compute_tokens,
        },
    }


def _check(rows, lease_stock, lease_proposed, identity) -> None:
    for row in rows:
        n_tokens = row["prompt_tokens"]
        if row["scenario"] == "stale":
            if row["stock_a_compute_tokens"] != n_tokens:
                raise SystemExit(f"stale stock A compute != prompt ({row})")
            if row["proposed_a_compute_tokens"] != 0:
                raise SystemExit(f"stale proposed A compute != 0 ({row})")
            if row["proposed_miss_reason"] != "evicted":
                raise SystemExit(f"stale miss reason ({row})")
        if row["scenario"] == "partial":
            if row["stock_a_compute_tokens"] != n_tokens // 2:
                raise SystemExit(f"partial stock A compute ({row})")
            if row["proposed_a_compute_tokens"] != 0:
                raise SystemExit(f"partial proposed A compute ({row})")
            if row["proposed_pulled_tokens"] != n_tokens // 2:
                raise SystemExit(f"partial pull length ({row})")
        if row["scenario"] == "hit":
            if row["proposed_a_compute_tokens"] != 0:
                raise SystemExit(f"hit proposed A compute ({row})")
            if row["stock_a_compute_tokens"] != BLOCK_SIZE:
                raise SystemExit(f"hit stock A compute ({row})")
    if lease_stock.stale_reads != 1 or lease_stock.lease_refuses != 0:
        raise SystemExit("stock lease path did not stale-read")
    if lease_proposed.lease_refuses != 1 or lease_proposed.stale_reads != 0:
        raise SystemExit("proposed lease path did not refuse")
    if lease_proposed.a_compute_tokens != 0 or identity.a_compute_tokens != 0:
        raise SystemExit("lease/identity scheduled compute on A")
    if identity.miss_reason != "identity_mismatch" or identity.pulled_tokens != 0:
        raise SystemExit("identity mismatch pulled blocks")


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RFC 65967 microbench (CPU-only mock)",
        "",
        "No GPU, no NIXL byte transfer, no model weights.",
        "A/B compute columns are exact scheduled prefill tokens.",
        "TTFT columns are p50 localhost TCP control-plane time in microseconds,",
        "including JSON of the token ids, over "
        + str(report["iterations"])
        + " iterations. They are not GPU time-to-first-token.",
        "The lookup column is in-process lookup-and-pin for that scenario,",
        "with no block hashing and no TCP.",
        "",
        "| scenario | tokens | stock A | proposed A | stock B | proposed B | "
        "stock critical | proposed critical | stock TTFT us | proposed TTFT us | "
        "lookup us |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {scenario} | {prompt_tokens} | {stock_a_compute_tokens} | "
            "{proposed_a_compute_tokens} | {stock_b_compute_tokens} | "
            "{proposed_b_compute_tokens} | {stock_critical_path_tokens} | "
            "{proposed_critical_path_tokens} | {stock_ttft} | {proposed_ttft} | "
            "{lookup} |".format(
                stock_ttft=_fmt(row["stock_b_ttft_control_us_p50"]),
                proposed_ttft=_fmt(row["proposed_b_ttft_control_us_p50"]),
                lookup=_fmt(row["lookup_and_pin_us_p50"]),
                **row,
            )
        )
    lease = report["lease_1024"]
    ident = report["identity_1024"]
    lines.extend(
        [
            "",
            "## Lease expiry at 1024 tokens",
            "",
            "- Stock blind read after recycle: "
            f"stale_reads={lease['stock_stale_reads']}, "
            f"refuses={lease['stock_lease_refuses']}, "
            f"pulled={lease['stock_pulled_tokens']}.",
            f"- Proposed: refuses={lease['proposed_lease_refuses']}, "
            f"stale_reads={lease['proposed_stale_reads']}, "
            f"A compute={lease['proposed_a_compute_tokens']}, "
            f"pulled={lease['proposed_pulled_tokens']}, "
            f"B recomputes {lease['proposed_b_compute_tokens']} tokens.",
            "",
            "## Identity mismatch at 1024 tokens",
            "",
            f"- miss_reason={ident['miss_reason']}, "
            f"A compute={ident['a_compute_tokens']}, "
            f"pulled={ident['pulled_tokens']}, B compute={ident['b_compute_tokens']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render_markdown(report)
    (RESULTS_DIR / "bench.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
