# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-group prefix cache with touch-pins and a NIXL-style lease.

Rules copied from ``BlockPool`` / ``BlockHashToBlockMap`` that this RFC
depends on:

* A cached block with ``ref_cnt == 0`` sits in the free queue and may be
  evicted (hash removed, tombstone recorded, generation bumped).
* ``touch`` takes one reference and pulls a zero-ref block out of the free
  queue, which is what a prefix-cache hit does today.
* Dropping the last reference of a hashed block appends it (evict last).
* Lookup walks hashes in order and stops at the first miss. It never
  schedules compute.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from prototypes.rfc65967_prefix_lookup_pin.protocol import (
    HitKind,
    KvIdentity,
    LookupAndPinRequest,
    MissReason,
    PinReply,
    lease_params,
)


@dataclass
class _Block:
    block_id: int
    block_hash: bytes | None = None
    ref_cnt: int = 0
    generation: int = 0
    prev_free: int = -1
    next_free: int = -1
    in_free: bool = False


class _FreeList:
    """Doubly linked free queue. ``remove`` is O(1), as in ``FreeKVCacheBlockQueue``."""

    def __init__(self, blocks: list[_Block]):
        self.blocks = blocks
        self.head = 0 if blocks else -1
        self.tail = len(blocks) - 1
        self.num = len(blocks)
        for index, block in enumerate(blocks):
            block.prev_free = index - 1
            block.next_free = index + 1 if index + 1 < len(blocks) else -1
            block.in_free = True

    def __len__(self) -> int:
        return self.num

    def popleft(self) -> int:
        if self.head < 0:
            raise MemoryError("prefix index has no free blocks")
        block_id = self.head
        self.remove(block_id)
        return block_id

    def remove(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if not block.in_free:
            return
        prev_id = block.prev_free
        next_id = block.next_free
        if prev_id >= 0:
            self.blocks[prev_id].next_free = next_id
        else:
            self.head = next_id
        if next_id >= 0:
            self.blocks[next_id].prev_free = prev_id
        else:
            self.tail = prev_id
        block.prev_free = -1
        block.next_free = -1
        block.in_free = False
        self.num -= 1

    def append(self, block_id: int) -> None:
        self._link(block_id, at_head=False)

    def appendleft(self, block_id: int) -> None:
        self._link(block_id, at_head=True)

    def _link(self, block_id: int, *, at_head: bool) -> None:
        block = self.blocks[block_id]
        if block.in_free:
            raise RuntimeError(f"block {block_id} is already in the free queue")
        block.in_free = True
        if self.head < 0:
            self.head = self.tail = block_id
            block.prev_free = block.next_free = -1
        elif at_head:
            block.prev_free = -1
            block.next_free = self.head
            self.blocks[self.head].prev_free = block_id
            self.head = block_id
        else:
            block.next_free = -1
            block.prev_free = self.tail
            self.blocks[self.tail].next_free = block_id
            self.tail = block_id
        self.num += 1


@dataclass
class PinSnapshot:
    pin_id: str
    block_ids: tuple[int, ...]
    generations: tuple[int, ...]
    hashes: tuple[bytes, ...]
    deadline_ns: int
    extension_ns: int


class PrefixIndex:
    """Hash map plus lease table for one engine, one KV group."""

    def __init__(
        self,
        *,
        num_blocks: int,
        identity: KvIdentity,
        lease_duration_s: float = 30.0,
        now_ns: Callable[[], int],
        tombstone_cap: int = 8192,
        engine_id: str = "",
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.identity = identity
        self.block_size = identity.block_size
        self.lease_duration_s = lease_duration_s
        self.now_ns = now_ns
        self.engine_id = engine_id
        self.host = host
        self.port = port
        self.tombstone_cap = tombstone_cap
        self.blocks = [_Block(i) for i in range(num_blocks)]
        self.free = _FreeList(self.blocks)
        self.by_hash: dict[bytes, int] = {}
        self.tombstones: OrderedDict[bytes, None] = OrderedDict()
        self.pins: dict[str, PinSnapshot] = {}
        self.expired: set[str] = set()
        self.completed: set[str] = set()
        self.snapshots: dict[str, PinSnapshot] = {}

    def reset(self) -> None:
        num_blocks = len(self.blocks)
        self.blocks = [_Block(i) for i in range(num_blocks)]
        self.free = _FreeList(self.blocks)
        self.by_hash.clear()
        self.tombstones.clear()
        self.pins.clear()
        self.expired.clear()
        self.completed.clear()
        self.snapshots.clear()

    def insert_prefix(self, hashes: tuple[bytes, ...] | list[bytes]) -> list[bytes]:
        """Cache full blocks with ref_cnt 0. Returns hashes evicted for room."""

        evicted: list[bytes] = []
        for block_hash in hashes:
            if block_hash in self.by_hash:
                continue
            block = self._alloc(evicted)
            block.block_hash = block_hash
            self.by_hash[block_hash] = block.block_id
            block.ref_cnt = 0
            self.free.append(block.block_id)
        return evicted

    def evict_hashes(self, hashes: tuple[bytes, ...] | list[bytes]) -> list[bytes]:
        """Drop unpinned cached blocks. Pinned blocks are left in place."""

        removed: list[bytes] = []
        for block_hash in hashes:
            block_id = self.by_hash.get(block_hash)
            if block_id is None:
                continue
            block = self.blocks[block_id]
            if block.ref_cnt > 0:
                continue
            self._forget_hash(block)
            removed.append(block_hash)
        return removed

    def reuse_ids(
        self,
        block_ids: tuple[int, ...] | list[int],
        new_hashes: tuple[bytes, ...] | list[bytes],
    ) -> None:
        """Recycle physical ids. Refuses while a pin still holds a reference."""

        if len(block_ids) != len(new_hashes):
            raise ValueError("reuse_ids requires one new hash per block id")
        for block_id, new_hash in zip(block_ids, new_hashes):
            block = self.blocks[block_id]
            if block.ref_cnt != 0:
                raise RuntimeError(
                    f"block {block_id} is pinned (ref_cnt={block.ref_cnt})"
                )
            if block.block_hash is not None:
                self._forget_hash(block)
            else:
                block.generation += 1
            self._discard_free(block_id)
            block.block_hash = new_hash
            self.by_hash[new_hash] = block_id
            self.free.append(block_id)

    def leading_hit_blocks(self, hashes: tuple[bytes, ...] | list[bytes]) -> int:
        hit = 0
        for block_hash in hashes:
            if block_hash not in self.by_hash:
                break
            hit += 1
        return hit

    def lookup_and_pin(self, request: LookupAndPinRequest) -> PinReply:
        """Longest cached run, pinned. Does not schedule prefill tokens."""

        if request.kv_identity != self.identity:
            return self._miss(MissReason.IDENTITY_MISMATCH, request.kv_identity)
        if not request.block_hashes:
            return self._miss(MissReason.NEVER_CACHED, self.identity)

        hit_ids: list[int] = []
        miss = MissReason.NONE
        for block_hash in request.block_hashes:
            block_id = self.by_hash.get(block_hash)
            if block_id is None:
                miss = (
                    MissReason.EVICTED
                    if block_hash in self.tombstones
                    else MissReason.NEVER_CACHED
                )
                break
            hit_ids.append(block_id)
        if not hit_ids:
            return self._miss(miss, self.identity)

        for block_id in hit_ids:
            self._touch(self.blocks[block_id])

        duration_s = (
            self.lease_duration_s
            if request.lease_duration_s is None
            else request.lease_duration_s
        )
        duration_ns, extension_ns, interval_ns = lease_params(duration_s)
        deadline = self.now_ns() + duration_ns
        pin_id = f"pin-{uuid4().hex}"
        snapshot = PinSnapshot(
            pin_id=pin_id,
            block_ids=tuple(hit_ids),
            generations=tuple(self.blocks[i].generation for i in hit_ids),
            hashes=tuple(request.block_hashes[: len(hit_ids)]),
            deadline_ns=deadline,
            extension_ns=extension_ns,
        )
        self.pins[pin_id] = snapshot
        self.snapshots[pin_id] = snapshot
        full = len(hit_ids) == len(request.block_hashes)
        return PinReply(
            hit_kind=HitKind.FULL if full else HitKind.PARTIAL,
            hit_tokens=len(hit_ids) * self.block_size,
            hit_blocks=len(hit_ids),
            remote_block_ids=(tuple(hit_ids),),
            pin_id=pin_id,
            engine_id=self.engine_id,
            host=self.host,
            port=self.port,
            lease_deadline_unix_ns=deadline,
            lease_extension_ns=extension_ns,
            heartbeat_interval_ns=interval_ns,
            miss_reason=MissReason.NONE if full else miss,
            kv_identity=self.identity,
        )

    def release_expired(self, now: int | None = None) -> int:
        """Drop pins whose deadline has passed. Returns how many were dropped."""

        now = self.now_ns() if now is None else now
        released = 0
        for pin_id, snapshot in list(self.pins.items()):
            if now >= snapshot.deadline_ns:
                self._drop_pin(snapshot, expired=True)
                released += 1
        return released

    def heartbeat(self, pin_id: str, now: int | None = None) -> int | None:
        """Extend a live lease. Returns the new deadline, or None if it is gone.

        Extension uses ``max(old, now + extension)``, same as
        ``NixlConnector`` heartbeat handling, so a late HB cannot shorten it.
        """

        now = self.now_ns() if now is None else now
        self.release_expired(now)
        snapshot = self.pins.get(pin_id)
        if snapshot is None:
            return None
        snapshot.deadline_ns = max(snapshot.deadline_ns, now + snapshot.extension_ns)
        return snapshot.deadline_ns

    def complete_pull(self, pin_id: str) -> bool:
        """Producer-side free on read completion, before the lease deadline."""

        snapshot = self.pins.get(pin_id)
        if snapshot is None:
            return False
        self._drop_pin(snapshot, expired=False)
        return True

    def authorize_pull(self, pin_id: str, now: int | None = None) -> str:
        """Return ``ok``, ``lease_expired``, or ``unknown``.

        Sweeps expired pins first, so a late reader observes the same state
        as a producer that has already logged "Releasing expired KV blocks".
        """

        now = self.now_ns() if now is None else now
        self.release_expired(now)
        snapshot = self.pins.get(pin_id)
        if snapshot is None:
            if pin_id in self.expired:
                return MissReason.LEASE_EXPIRED.value
            return "unknown"
        if not self.snapshot_intact(snapshot):
            return MissReason.LEASE_EXPIRED.value
        return "ok"

    def snapshot_intact(self, snapshot: PinSnapshot) -> bool:
        for block_id, generation, block_hash in zip(
            snapshot.block_ids, snapshot.generations, snapshot.hashes
        ):
            block = self.blocks[block_id]
            if block.generation != generation or block.block_hash != block_hash:
                return False
        return True

    def blind_read_intact(self, pin_id: str) -> bool:
        """Stock pull: trust the block ids in the original reply.

        True when those physical blocks still hold the pinned generation.
        False after expiry-and-recycle (the #66009 silent stale read).
        """

        snapshot = self.snapshots.get(pin_id)
        if snapshot is None:
            return False
        return self.snapshot_intact(snapshot)

    def _miss(self, reason: MissReason, identity: KvIdentity) -> PinReply:
        return PinReply(
            hit_kind=HitKind.MISS,
            hit_tokens=0,
            hit_blocks=0,
            remote_block_ids=(),
            pin_id="",
            engine_id=self.engine_id,
            host=self.host,
            port=self.port,
            lease_deadline_unix_ns=0,
            lease_extension_ns=0,
            heartbeat_interval_ns=0,
            miss_reason=reason,
            kv_identity=identity,
        )

    def _alloc(self, evicted: list[bytes]) -> _Block:
        block_id = self.free.popleft()
        block = self.blocks[block_id]
        if block.ref_cnt != 0:
            raise RuntimeError(f"free queue contained pinned block {block_id}")
        if block.block_hash is not None:
            evicted.append(block.block_hash)
            self._forget_hash(block)
        return block

    def _forget_hash(self, block: _Block) -> None:
        block_hash = block.block_hash
        if block_hash is None:
            return
        self.tombstones[block_hash] = None
        self.tombstones.move_to_end(block_hash)
        while len(self.tombstones) > self.tombstone_cap:
            self.tombstones.popitem(last=False)
        self.by_hash.pop(block_hash, None)
        block.block_hash = None
        block.generation += 1

    def _touch(self, block: _Block) -> None:
        if block.ref_cnt == 0:
            self._discard_free(block.block_id)
        block.ref_cnt += 1

    def _unpin(self, block: _Block) -> None:
        if block.ref_cnt <= 0:
            raise RuntimeError(f"unpin of block {block.block_id} with ref 0")
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            if block.block_hash is None:
                self.free.appendleft(block.block_id)
            else:
                self.free.append(block.block_id)

    def _discard_free(self, block_id: int) -> None:
        self.free.remove(block_id)

    def _drop_pin(self, snapshot: PinSnapshot, *, expired: bool) -> None:
        for block_id in snapshot.block_ids:
            self._unpin(self.blocks[block_id])
        self.pins.pop(snapshot.pin_id, None)
        if expired:
            self.expired.add(snapshot.pin_id)
        else:
            self.completed.add(snapshot.pin_id)
