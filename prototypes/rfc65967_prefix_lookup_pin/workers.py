# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two workers sharing a prefix-hash index. No GPU, no NIXL read."""

from __future__ import annotations

from typing import Callable, Sequence

from prototypes.rfc65967_prefix_lookup_pin.prefix_index import PrefixIndex
from prototypes.rfc65967_prefix_lookup_pin.protocol import (
    KvIdentity,
    LookupAndPinRequest,
    PinReply,
    block_hashes,
    scheduled_prefill_tokens,
)


class RouterIndex:
    """Eventually consistent hash -> engine map, fed by store/remove events.

    ``hold_removals`` keeps ``BlockRemoved`` queued, which is the stale
    index in RFC #65967: the router still points at A after A has evicted.
    """

    def __init__(self) -> None:
        self.locations: dict[bytes, str] = {}
        self.hold_removals = False
        self.pending_removals: list[bytes] = []

    def reset(self) -> None:
        self.locations.clear()
        self.hold_removals = False
        self.pending_removals.clear()

    def stored(self, engine_id: str, hashes: Sequence[bytes]) -> None:
        for block_hash in hashes:
            self.locations[block_hash] = engine_id

    def removed(self, hashes: Sequence[bytes]) -> None:
        if self.hold_removals:
            self.pending_removals.extend(hashes)
            return
        for block_hash in hashes:
            self.locations.pop(block_hash, None)

    def prefix_blocks(self, hashes: Sequence[bytes]) -> int:
        hit = 0
        for block_hash in hashes:
            if block_hash not in self.locations:
                break
            hit += 1
        return hit

    def owner(self, hashes: Sequence[bytes]) -> str | None:
        if self.prefix_blocks(hashes) == 0:
            return None
        return self.locations.get(hashes[0])


class Worker:
    """One engine's prefix cache and the requests it actually schedules."""

    def __init__(
        self,
        engine_id: str,
        identity: KvIdentity,
        *,
        router: RouterIndex,
        now_ns: Callable[[], int],
        num_blocks: int = 8192,
        lease_duration_s: float = 30.0,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.engine_id = engine_id
        self.identity = identity
        self.router = router
        self.now_ns = now_ns
        self.index = PrefixIndex(
            num_blocks=num_blocks,
            identity=identity,
            lease_duration_s=lease_duration_s,
            now_ns=now_ns,
            engine_id=engine_id,
            host=host,
            port=port,
        )
        self.compute_tokens = 0
        self.lease_refuses = 0
        self.stale_reads = 0
        self.lookups = 0

    def set_endpoint(self, host: str, port: int) -> None:
        self.index.host = host
        self.index.port = port

    def reset_cache(self) -> None:
        self.index.reset()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.compute_tokens = 0
        self.lease_refuses = 0
        self.stale_reads = 0
        self.lookups = 0

    def hashes_for(self, token_ids: Sequence[int]) -> tuple[bytes, ...]:
        return block_hashes(token_ids, self.identity)

    def cache_prefix(self, token_ids: Sequence[int]) -> tuple[bytes, ...]:
        hashes = self.hashes_for(token_ids)
        evicted = self.index.insert_prefix(hashes)
        self.router.stored(self.engine_id, hashes)
        self.router.removed(evicted)
        return hashes

    def evict_prefix(self, token_ids: Sequence[int]) -> list[bytes]:
        removed = self.index.evict_hashes(self.hashes_for(token_ids))
        self.router.removed(removed)
        return removed

    def note_compute(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("compute tokens must be >= 0")
        self.compute_tokens += tokens

    def stock_probe(self, token_ids: Sequence[int]) -> PinReply:
        """Router step 1: admit the prefix on A and prefill whatever missed.

        A full hit still schedules the last block (logits + block alignment).
        A miss schedules the whole prefix. That scheduled work is the cost
        the RFC wants to remove; this method is the only one that adds it.
        """

        hashes = self.hashes_for(token_ids)
        cached_tokens = self.index.leading_hit_blocks(hashes) * self.identity.block_size
        compute = scheduled_prefill_tokens(
            len(token_ids), cached_tokens, self.identity.block_size
        )
        self.note_compute(compute)
        if self.index.leading_hit_blocks(hashes) < len(hashes):
            evicted = self.index.insert_prefix(hashes)
            self.router.stored(self.engine_id, hashes)
            self.router.removed(evicted)
        self.lookups += 1
        return self.index.lookup_and_pin(
            LookupAndPinRequest(block_hashes=hashes, kv_identity=self.identity)
        )

    def lookup_raw(
        self,
        hashes: tuple[bytes, ...],
        identity: KvIdentity,
        lease_duration_s: float | None = None,
    ) -> PinReply:
        """Compute-free lookup-and-pin. Does not increment ``compute_tokens``."""

        self.lookups += 1
        return self.index.lookup_and_pin(
            LookupAndPinRequest(
                block_hashes=hashes,
                kv_identity=identity,
                lease_duration_s=lease_duration_s,
            )
        )

    def install_pulled(self, hashes: Sequence[bytes]) -> None:
        evicted = self.index.insert_prefix(tuple(hashes))
        self.router.stored(self.engine_id, hashes)
        self.router.removed(evicted)

    def local_generate(self, token_ids: Sequence[int]) -> int:
        """Schedule whatever the local prefix cache did not cover."""

        hashes = self.hashes_for(token_ids)
        cached = self.index.leading_hit_blocks(hashes) * self.identity.block_size
        compute = scheduled_prefill_tokens(
            len(token_ids), cached, self.identity.block_size
        )
        self.note_compute(compute)
        if self.index.leading_hit_blocks(hashes) < len(hashes):
            self.install_pulled(hashes)
        return compute

    def stock_warmup(
        self,
        token_ids: Sequence[int],
        reply: PinReply,
        producer: Worker,
    ) -> bool:
        """Router step 2. Trust the block ids. Do not check the lease.

        Returns whether the physical blocks still match the pin. A false
        return is a stale read that stock still installs.
        """

        intact = True
        if reply.pin_id:
            intact = producer.index.blind_read_intact(reply.pin_id)
            if not intact:
                self.stale_reads += 1
        if reply.hit_blocks > 0:
            hashes = self.hashes_for(token_ids)[: reply.hit_blocks]
            self.install_pulled(hashes)
            producer.index.complete_pull(reply.pin_id)
        return intact

    def pull_pinned(
        self,
        token_ids: Sequence[int],
        reply: PinReply,
        producer: Worker,
    ) -> str:
        """Proposed pull: refuse an expired lease, otherwise install the hit.

        Returns ``ok``, ``lease_expired``, ``miss``, or ``identity_mismatch``.
        """

        if reply.miss_reason.value == "identity_mismatch":
            return "identity_mismatch"
        if reply.hit_tokens <= 0 or not reply.pin_id:
            return "miss"
        status = producer.index.authorize_pull(reply.pin_id)
        if status != "ok":
            self.lease_refuses += 1
            return status
        hashes = block_hashes(token_ids, reply.kv_identity)[: reply.hit_blocks]
        self.install_pulled(hashes)
        producer.index.complete_pull(reply.pin_id)
        return "ok"
