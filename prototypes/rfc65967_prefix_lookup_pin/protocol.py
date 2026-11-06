# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire contract for compute-free prefix lookup-and-pin.

The dataclasses match the kv_transfer_params / side-channel shapes sketched
in DESIGN.md. They are the contract a NixlConnector patch would encode with
msgspec next to ``GET_META_MSG``; this module stays importable without vLLM.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence


class MissReason(str, Enum):
    """Why a lookup stopped, or why a later pull was refused.

    ``LEASE_EXPIRED`` is a pull-time result. A pin reply uses the other
    values: ``NONE`` when every requested full block was pinned, otherwise
    the reason the walk stopped. A partial hit has ``hit_tokens > 0`` and a
    non-``NONE`` reason for the next block. Callers must read ``hit_kind``
    before treating the reason as a total miss.
    """

    NONE = "none"
    NEVER_CACHED = "never_cached"
    EVICTED = "evicted"
    IDENTITY_MISMATCH = "identity_mismatch"
    LEASE_EXPIRED = "lease_expired"


class HitKind(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    MISS = "miss"


@dataclass(frozen=True)
class KvIdentity:
    """Fail-closed identity for a peer pull.

    ``compatibility_hash`` is ``compute_nixl_compatibility_hash`` (model,
    dtype, KV heads, connector version, transfer mode). It is not part of
    the block hash: two models with the same tokenizer produce the same
    token-block hashes and must still refuse to pull.

    ``cache_salt`` is the per-tenant isolation key mixed into the first
    block hash by ``hash_block_tokens`` / ``generate_block_hash_extra_keys``.
    Declaring a different salt than the producer fails closed even when the
    caller computed hashes under the producer's salt.
    """

    compatibility_hash: str
    cache_salt: str
    block_size: int
    hash_algo: str = "sha256"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> KvIdentity:
        return cls(
            compatibility_hash=payload["compatibility_hash"],
            cache_salt=payload["cache_salt"],
            block_size=int(payload["block_size"]),
            hash_algo=payload.get("hash_algo", "sha256"),
        )


@dataclass(frozen=True)
class LookupAndPinRequest:
    """Connector-to-connector lookup. Hashes are full blocks, prefix order."""

    block_hashes: tuple[bytes, ...]
    kv_identity: KvIdentity
    lease_duration_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "block_hashes": [h.hex() for h in self.block_hashes],
            "kv_identity": self.kv_identity.to_json(),
            "lease_duration_s": self.lease_duration_s,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> LookupAndPinRequest:
        return cls(
            block_hashes=tuple(bytes.fromhex(h) for h in payload["block_hashes"]),
            kv_identity=KvIdentity.from_json(payload["kv_identity"]),
            lease_duration_s=payload.get("lease_duration_s"),
        )


@dataclass
class PinReply:
    """What A returns, and what B copies into kv_transfer_params.

    ``lease_deadline_unix_ns`` is wall time (``time.time_ns``), not
    ``time.perf_counter``. perf_counter epochs differ across processes
    (see NixlConnector's scheduler_clock rebase and issue #66009).
    ``lease_extension_ns`` is the heartbeat renewal quantum
    (``kv_lease_duration * 2/3``) so B can advance its local deadline
    model when a heartbeat is accepted.
    """

    hit_kind: HitKind
    hit_tokens: int
    hit_blocks: int
    remote_block_ids: tuple[tuple[int, ...], ...]
    pin_id: str
    engine_id: str
    host: str
    port: int
    lease_deadline_unix_ns: int
    lease_extension_ns: int
    heartbeat_interval_ns: int
    miss_reason: MissReason
    kv_identity: KvIdentity
    transfer_mode: str = "pull"

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hit_kind"] = self.hit_kind.value
        payload["miss_reason"] = self.miss_reason.value
        payload["remote_block_ids"] = [list(group) for group in self.remote_block_ids]
        payload["kv_identity"] = self.kv_identity.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> PinReply:
        return cls(
            hit_kind=HitKind(payload["hit_kind"]),
            hit_tokens=int(payload["hit_tokens"]),
            hit_blocks=int(payload["hit_blocks"]),
            remote_block_ids=tuple(
                tuple(group) for group in payload["remote_block_ids"]
            ),
            pin_id=payload["pin_id"],
            engine_id=payload["engine_id"],
            host=payload["host"],
            port=int(payload["port"]),
            lease_deadline_unix_ns=int(payload["lease_deadline_unix_ns"]),
            lease_extension_ns=int(payload["lease_extension_ns"]),
            heartbeat_interval_ns=int(payload["heartbeat_interval_ns"]),
            miss_reason=MissReason(payload["miss_reason"]),
            kv_identity=KvIdentity.from_json(payload["kv_identity"]),
            transfer_mode=payload.get("transfer_mode", "pull"),
        )


def lease_params(duration_s: float) -> tuple[int, int, int]:
    """Return ``(duration_ns, extension_ns, heartbeat_interval_ns)``.

    Matches NixlBaseConnectorScheduler: heartbeat interval is
    ``kv_lease_duration // 6`` and the extension applied on HB is
    ``kv_lease_duration * 2 // 3``.
    """

    if duration_s < 0:
        raise ValueError(f"lease duration must be >= 0, got {duration_s}")
    duration_ns = int(duration_s * 1_000_000_000)
    extension_ns = duration_ns * 2 // 3
    interval_ns = duration_ns // 6
    return duration_ns, extension_ns, interval_ns


# Stand-in for vLLM's NONE_HASH. The production hash chains
# ``hash_block_tokens(parent, token_ids, extra_keys)``; tests only need a
# stable chained hash so eviction and salt changes are observable.
NONE_HASH = hashlib.sha256(b"vllm-none-hash").digest()


def hash_block(
    parent: bytes,
    token_ids: tuple[int, ...],
    extra: tuple[Any, ...] | None,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(parent)
    digest.update(repr((token_ids, extra)).encode())
    return digest.digest()


def block_hashes(
    token_ids: Sequence[int],
    identity: KvIdentity,
) -> tuple[bytes, ...]:
    """Full-block chained hashes. The tail short of ``block_size`` is omitted.

    ``cache_salt`` is an extra key on the first block only, matching
    ``generate_block_hash_extra_keys``.
    """

    block_size = identity.block_size
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    usable = len(token_ids) - (len(token_ids) % block_size)
    parent = NONE_HASH
    hashes: list[bytes] = []
    for offset in range(0, usable, block_size):
        chunk = tuple(token_ids[offset : offset + block_size])
        extra = (identity.cache_salt,) if offset == 0 and identity.cache_salt else None
        parent = hash_block(parent, chunk, extra)
        hashes.append(parent)
    return tuple(hashes)


def scheduled_prefill_tokens(
    prompt_tokens: int,
    cached_tokens: int,
    block_size: int,
) -> int:
    """Tokens a stock prefill schedules given a block-aligned cache hit.

    Mirrors ``KVCacheManager.get_computed_blocks``: the last prompt token is
    always recomputed so the model can emit logits, and the hit is floored
    to a block boundary. A block-aligned full hit therefore recomputes one
    whole block, not one token.
    """

    if prompt_tokens <= 0:
        return 0
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    max_hit = prompt_tokens - 1
    hit = min(max(cached_tokens, 0), max_hit)
    hit -= hit % block_size
    return prompt_tokens - hit
