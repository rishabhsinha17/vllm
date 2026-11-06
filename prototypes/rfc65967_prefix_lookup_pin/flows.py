# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stock 3-request peer reuse versus one request on B that pulls from A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from prototypes.rfc65967_prefix_lookup_pin.nixl_hooks import (
    decline_expired_lease,
    peer_remote_token_count,
)
from prototypes.rfc65967_prefix_lookup_pin.protocol import (
    KvIdentity,
    MissReason,
    PinReply,
    block_hashes,
    scheduled_prefill_tokens,
)
from prototypes.rfc65967_prefix_lookup_pin.workers import RouterIndex, Worker


@dataclass
class FlowOutcome:
    """One executed flow. Token counts are exact; times are added by the bench."""

    flow: str
    scenario: str
    prompt_tokens: int
    a_compute_tokens: int
    b_compute_tokens: int
    pulled_tokens: int
    http_requests: int
    sidechannel_rpcs: int
    lease_refuses: int
    stale_reads: int
    miss_reason: str
    hit_kind: str
    hit_tokens: int
    router_owner: str | None
    external_tokens: int


class Clock:
    """Injectable unix-ns clock. Production code uses ``time.time_ns``."""

    def __init__(self, start: int = 1_700_000_000_000_000_000) -> None:
        self.ns = start

    def now(self) -> int:
        return self.ns

    def advance(self, delta_ns: int) -> None:
        self.ns += delta_ns

    def advance_s(self, seconds: float) -> None:
        self.advance(int(seconds * 1_000_000_000) + 1)


@dataclass
class Cluster:
    A: Worker
    B: Worker
    router: RouterIndex
    identity: KvIdentity
    lease_s: float
    clock: Clock

    def reset(self) -> None:
        self.router.reset()
        self.A.reset_cache()
        self.B.reset_cache()


def make_cluster(
    *,
    clock: Clock | None = None,
    block_size: int = 16,
    lease_s: float = 30.0,
    num_blocks: int = 8192,
    compatibility_hash: str = "nixl-compat-qwen",
    cache_salt: str = "tenant-a",
) -> Cluster:
    clock = clock or Clock()
    identity = KvIdentity(
        compatibility_hash=compatibility_hash,
        cache_salt=cache_salt,
        block_size=block_size,
    )
    router = RouterIndex()
    worker_a = Worker(
        "engine-A",
        identity,
        router=router,
        now_ns=clock.now,
        num_blocks=num_blocks,
        lease_duration_s=lease_s,
        port=5557,
    )
    worker_b = Worker(
        "engine-B",
        identity,
        router=router,
        now_ns=clock.now,
        num_blocks=num_blocks,
        lease_duration_s=lease_s,
        port=5558,
    )
    return Cluster(worker_a, worker_b, router, identity, lease_s, clock)


def prepare(cluster: Cluster, scenario: str, token_ids: Sequence[int]) -> None:
    """Seed A and the router. Does not run a request."""

    cluster.reset()
    block = cluster.identity.block_size
    if scenario == "stale":
        cluster.A.cache_prefix(token_ids)
        cluster.router.hold_removals = True
        cluster.A.evict_prefix(token_ids)
        return
    if scenario == "hit":
        cluster.A.cache_prefix(token_ids)
        return
    if scenario == "partial":
        if len(token_ids) < block * 2 or len(token_ids) % (block * 2):
            raise ValueError("partial scenario needs an even block-aligned prompt")
        cluster.A.cache_prefix(token_ids[: len(token_ids) // 2])
        return
    if scenario in ("identity", "lease"):
        cluster.A.cache_prefix(token_ids)
        return
    raise ValueError(f"unknown scenario {scenario}")


def _delta(worker: Worker, before_compute: int, before_refuses: int, before_stale: int):
    return (
        worker.compute_tokens - before_compute,
        worker.lease_refuses - before_refuses,
        worker.stale_reads - before_stale,
    )


def run_stock(cluster: Cluster, scenario: str, token_ids: Sequence[int]) -> FlowOutcome:
    """Three HTTP requests: probe A, warmup pull on B, real request on B.

    All-or-nothing: the probe prefills whatever A does not have, so B pulls
    a full prefix. The warmup does not look at the lease deadline.
    """

    if scenario == "lease":
        return _run_stock_lease(cluster, token_ids)
    hashes = cluster.A.hashes_for(token_ids)
    owner = cluster.router.owner(hashes)
    a0, ar0, as0 = (
        cluster.A.compute_tokens,
        cluster.A.lease_refuses,
        cluster.A.stale_reads,
    )
    b0, br0, bs0 = (
        cluster.B.compute_tokens,
        cluster.B.lease_refuses,
        cluster.B.stale_reads,
    )
    reply = cluster.A.stock_probe(list(token_ids))
    cluster.B.stock_warmup(token_ids, reply, cluster.A)
    cluster.B.local_generate(token_ids)
    a_compute, _, _ = _delta(cluster.A, a0, ar0, as0)
    b_compute, b_refuses, b_stale = _delta(cluster.B, b0, br0, bs0)
    external, _ = peer_remote_token_count(
        do_remote_prefill=True,
        prompt_tokens=len(token_ids),
        num_computed_tokens=0,
        remote_num_tokens=reply.hit_tokens,
        peer_lookup_resolved=False,
        prefill_backoff=0,
    )
    return FlowOutcome(
        flow="stock",
        scenario=scenario,
        prompt_tokens=len(token_ids),
        a_compute_tokens=a_compute,
        b_compute_tokens=b_compute,
        pulled_tokens=reply.hit_tokens,
        http_requests=3,
        sidechannel_rpcs=0,
        lease_refuses=b_refuses,
        stale_reads=b_stale,
        miss_reason=reply.miss_reason.value,
        hit_kind=reply.hit_kind.value,
        hit_tokens=reply.hit_tokens,
        router_owner=owner,
        external_tokens=external,
    )


def run_proposed(
    cluster: Cluster,
    scenario: str,
    token_ids: Sequence[int],
    *,
    identity: KvIdentity | None = None,
) -> FlowOutcome:
    """One request to B. B pins on A, pulls the hit, prefills the rest."""

    if scenario == "lease":
        return _run_proposed_lease(cluster, token_ids)

    hashes = block_hashes(token_ids, cluster.identity)
    owner = cluster.router.owner(hashes)
    request_identity = identity or cluster.identity
    if scenario == "identity":
        request_identity = KvIdentity(
            compatibility_hash=cluster.identity.compatibility_hash + "-other-model",
            cache_salt=cluster.identity.cache_salt,
            block_size=cluster.identity.block_size,
        )
    a0 = cluster.A.compute_tokens
    b0, br0, bs0 = (
        cluster.B.compute_tokens,
        cluster.B.lease_refuses,
        cluster.B.stale_reads,
    )
    reply = cluster.A.lookup_raw(hashes, request_identity)
    pull_status = cluster.B.pull_pinned(token_ids, reply, cluster.A)
    if pull_status != "ok":
        # No remote blocks. local_generate computes the prompt.
        pass
    external, _ = peer_remote_token_count(
        do_remote_prefill=True,
        prompt_tokens=len(token_ids),
        num_computed_tokens=0,
        remote_num_tokens=reply.hit_tokens if pull_status == "ok" else 0,
        peer_lookup_resolved=True,
        prefill_backoff=0,
    )
    # B's local prefill is the suffix after the external prefix, with the
    # same last-block rule as the scheduler.
    cached_for_b = external
    b_compute_expected = scheduled_prefill_tokens(
        len(token_ids), cached_for_b, cluster.identity.block_size
    )
    # Drive it through B's cache so the counter comes from the worker.
    # pull_pinned already installed the hit; local_generate schedules the rest.
    cluster.B.local_generate(token_ids)
    b_compute, b_refuses, b_stale = _delta(cluster.B, b0, br0, bs0)
    if b_compute != b_compute_expected:
        raise RuntimeError(
            f"B compute {b_compute} != planned {b_compute_expected} ({scenario})"
        )
    pulled = reply.hit_tokens if pull_status == "ok" else 0
    return FlowOutcome(
        flow="proposed",
        scenario=scenario,
        prompt_tokens=len(token_ids),
        a_compute_tokens=cluster.A.compute_tokens - a0,
        b_compute_tokens=b_compute,
        pulled_tokens=pulled,
        http_requests=1,
        sidechannel_rpcs=1,
        lease_refuses=b_refuses,
        stale_reads=b_stale,
        miss_reason=reply.miss_reason.value,
        hit_kind=reply.hit_kind.value,
        hit_tokens=reply.hit_tokens,
        router_owner=owner,
        external_tokens=external,
    )


def _pin_for_lease(cluster: Cluster, token_ids: Sequence[int]) -> PinReply:
    hashes = cluster.A.hashes_for(token_ids)
    return cluster.A.lookup_raw(hashes, cluster.identity)


def _sweep_and_recycle(
    cluster: Cluster, reply: PinReply, token_ids: Sequence[int]
) -> None:
    """Producer drops the lease, then a later request reuses those block ids."""

    cluster.clock.advance_s(cluster.lease_s)
    released = cluster.A.index.release_expired()
    if released != 1:
        raise RuntimeError(f"expected to release 1 pin, released {released}")
    block_ids = reply.remote_block_ids[0]
    replacement = [token + 100_000 for token in token_ids]
    new_hashes = cluster.A.hashes_for(replacement)[: len(block_ids)]
    cluster.A.index.reuse_ids(block_ids, new_hashes)


def _run_stock_lease(cluster: Cluster, token_ids: Sequence[int]) -> FlowOutcome:
    """Pin, expire, recycle, then the stock warmup reads the stale ids."""

    a0 = cluster.A.compute_tokens
    b0, br0, bs0 = (
        cluster.B.compute_tokens,
        cluster.B.lease_refuses,
        cluster.B.stale_reads,
    )
    reply = _pin_for_lease(cluster, token_ids)
    _sweep_and_recycle(cluster, reply, token_ids)
    # Stock ignores the deadline (None-equivalent: it never consults it).
    declined = decline_expired_lease(None, cluster.A.now_ns())
    if declined:
        raise RuntimeError("stock path must not decline a missing deadline")
    cluster.B.stock_warmup(token_ids, reply, cluster.A)
    cluster.B.local_generate(token_ids)
    b_compute, b_refuses, b_stale = _delta(cluster.B, b0, br0, bs0)
    return FlowOutcome(
        flow="stock",
        scenario="lease",
        prompt_tokens=len(token_ids),
        a_compute_tokens=cluster.A.compute_tokens - a0,
        b_compute_tokens=b_compute,
        pulled_tokens=reply.hit_tokens,
        http_requests=3,
        sidechannel_rpcs=0,
        lease_refuses=b_refuses,
        stale_reads=b_stale,
        miss_reason=MissReason.LEASE_EXPIRED.value,
        hit_kind=reply.hit_kind.value,
        hit_tokens=reply.hit_tokens,
        router_owner=cluster.router.owner(cluster.A.hashes_for(token_ids)),
        external_tokens=len(token_ids),
    )


def _run_proposed_lease(cluster: Cluster, token_ids: Sequence[int]) -> FlowOutcome:
    """Same pin and recycle. B refuses because the unix deadline has passed."""

    a0 = cluster.A.compute_tokens
    b0, br0, bs0 = (
        cluster.B.compute_tokens,
        cluster.B.lease_refuses,
        cluster.B.stale_reads,
    )
    reply = _pin_for_lease(cluster, token_ids)
    deadline = reply.lease_deadline_unix_ns
    _sweep_and_recycle(cluster, reply, token_ids)
    if not decline_expired_lease(deadline, cluster.A.now_ns()):
        raise RuntimeError("proposed path should see an expired unix deadline")
    status = cluster.B.pull_pinned(token_ids, reply, cluster.A)
    if status != MissReason.LEASE_EXPIRED.value:
        raise RuntimeError(f"expected lease_expired, got {status}")
    cluster.B.local_generate(token_ids)
    b_compute, b_refuses, b_stale = _delta(cluster.B, b0, br0, bs0)
    return FlowOutcome(
        flow="proposed",
        scenario="lease",
        prompt_tokens=len(token_ids),
        a_compute_tokens=cluster.A.compute_tokens - a0,
        b_compute_tokens=b_compute,
        pulled_tokens=0,
        http_requests=1,
        sidechannel_rpcs=1,
        lease_refuses=b_refuses,
        stale_reads=b_stale,
        miss_reason=MissReason.LEASE_EXPIRED.value,
        hit_kind=reply.hit_kind.value,
        hit_tokens=reply.hit_tokens,
        router_owner=cluster.router.owner(cluster.A.hashes_for(token_ids)),
        external_tokens=0,
    )
