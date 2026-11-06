# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for the RFC #65967 lookup-and-pin prototype."""

from __future__ import annotations

import time
import unittest

from prototypes.rfc65967_prefix_lookup_pin.flows import (
    Clock,
    make_cluster,
    prepare,
    run_proposed,
    run_stock,
)
from prototypes.rfc65967_prefix_lookup_pin.nixl_hooks import (
    decline_expired_lease,
    peer_remote_token_count,
)
from prototypes.rfc65967_prefix_lookup_pin.prefix_index import PrefixIndex
from prototypes.rfc65967_prefix_lookup_pin.protocol import (
    KvIdentity,
    LookupAndPinRequest,
    MissReason,
    block_hashes,
    lease_params,
    scheduled_prefill_tokens,
)


def _identity(block_size: int = 16, salt: str = "tenant") -> KvIdentity:
    return KvIdentity(
        compatibility_hash="compat-a",
        cache_salt=salt,
        block_size=block_size,
    )


def _tokens(n: int, seed: int = 0) -> list[int]:
    return list(range(seed, seed + n))


class TestProtocol(unittest.TestCase):
    def test_lease_quantum_matches_nixl(self) -> None:
        duration_ns, extension_ns, interval_ns = lease_params(30)
        self.assertEqual(duration_ns, 30_000_000_000)
        self.assertEqual(extension_ns, 20_000_000_000)
        self.assertEqual(interval_ns, 5_000_000_000)

    def test_full_hit_recomputes_one_block(self) -> None:
        self.assertEqual(scheduled_prefill_tokens(256, 256, 16), 16)
        self.assertEqual(scheduled_prefill_tokens(256, 0, 16), 256)
        self.assertEqual(scheduled_prefill_tokens(256, 128, 16), 128)

    def test_salt_changes_the_block_hash(self) -> None:
        tokens = _tokens(32)
        left = block_hashes(tokens, _identity(salt="a"))
        right = block_hashes(tokens, _identity(salt="b"))
        self.assertNotEqual(left, right)
        self.assertEqual(len(left), 2)

    def test_partial_hook_uses_hit_length(self) -> None:
        count, load_async = peer_remote_token_count(
            do_remote_prefill=True,
            prompt_tokens=256,
            num_computed_tokens=0,
            remote_num_tokens=128,
            peer_lookup_resolved=True,
            prefill_backoff=0,
        )
        self.assertEqual((count, load_async), (128, True))

    def test_legacy_prefill_pulls_the_whole_prompt(self) -> None:
        count, load_async = peer_remote_token_count(
            do_remote_prefill=True,
            prompt_tokens=256,
            num_computed_tokens=0,
            remote_num_tokens=128,
            peer_lookup_resolved=False,
            prefill_backoff=0,
        )
        self.assertEqual((count, load_async), (256, True))

    def test_decline_requires_an_exported_deadline(self) -> None:
        self.assertFalse(decline_expired_lease(None, 10**18))
        self.assertFalse(decline_expired_lease(100, 99))
        self.assertTrue(decline_expired_lease(100, 100))
        self.assertTrue(decline_expired_lease(100, 90, safety_margin_ns=10))


class TestIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.identity = _identity()
        self.index = PrefixIndex(
            num_blocks=32,
            identity=self.identity,
            lease_duration_s=30.0,
            now_ns=self.clock.now,
            engine_id="engine-A",
        )

    def test_never_cached_does_not_pin(self) -> None:
        hashes = block_hashes(_tokens(64), self.identity)
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.assertEqual(reply.miss_reason, MissReason.NEVER_CACHED)
        self.assertEqual(reply.hit_tokens, 0)
        self.assertEqual(reply.remote_block_ids, ())
        self.assertEqual(sum(block.ref_cnt for block in self.index.blocks), 0)

    def test_full_hit_pins_without_a_compute_counter(self) -> None:
        hashes = block_hashes(_tokens(64), self.identity)
        self.index.insert_prefix(hashes)
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.assertEqual(reply.hit_kind.value, "full")
        self.assertEqual(reply.miss_reason, MissReason.NONE)
        self.assertEqual(reply.hit_tokens, 64)
        self.assertEqual(len(reply.remote_block_ids), 1)
        self.assertEqual(
            sum(block.ref_cnt for block in self.index.blocks), reply.hit_blocks
        )
        self.assertGreater(reply.lease_deadline_unix_ns, self.clock.now())
        self.assertEqual(reply.lease_extension_ns, 20_000_000_000)

    def test_evicted_is_not_never_cached(self) -> None:
        hashes = block_hashes(_tokens(32), self.identity)
        self.index.insert_prefix(hashes)
        self.assertEqual(self.index.evict_hashes(hashes), list(hashes))
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.assertEqual(reply.miss_reason, MissReason.EVICTED)
        self.assertEqual(reply.hit_kind.value, "miss")

    def test_partial_hit_and_suffix_reason(self) -> None:
        tokens = _tokens(64)
        hashes = block_hashes(tokens, self.identity)
        self.index.insert_prefix(hashes[:2])
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.assertEqual(reply.hit_kind.value, "partial")
        self.assertEqual(reply.hit_tokens, 32)
        self.assertEqual(reply.miss_reason, MissReason.NEVER_CACHED)

        self.index.complete_pull(reply.pin_id)
        self.index.insert_prefix(hashes)
        self.index.evict_hashes(hashes[2:])
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.assertEqual(reply.hit_tokens, 32)
        self.assertEqual(reply.miss_reason, MissReason.EVICTED)

    def test_identity_mismatch_fail_closed(self) -> None:
        hashes = block_hashes(_tokens(32), self.identity)
        self.index.insert_prefix(hashes)
        other = KvIdentity(
            compatibility_hash="compat-other",
            cache_salt=self.identity.cache_salt,
            block_size=self.identity.block_size,
        )
        reply = self.index.lookup_and_pin(LookupAndPinRequest(hashes, other))
        self.assertEqual(reply.miss_reason, MissReason.IDENTITY_MISMATCH)
        self.assertEqual(reply.hit_tokens, 0)
        self.assertEqual(reply.remote_block_ids, ())
        self.assertEqual(sum(block.ref_cnt for block in self.index.blocks), 0)

    def test_pin_is_not_evicted_under_pressure(self) -> None:
        tiny = PrefixIndex(
            num_blocks=4,
            identity=self.identity,
            now_ns=self.clock.now,
            engine_id="engine-A",
        )
        kept = block_hashes(_tokens(64), self.identity)
        tiny.insert_prefix(kept)
        reply = tiny.lookup_and_pin(LookupAndPinRequest(kept[:2], self.identity))
        extra = block_hashes(_tokens(64, seed=1000), self.identity)
        tiny.insert_prefix(extra)
        self.assertEqual(tiny.leading_hit_blocks(kept[:2]), 2)
        self.assertEqual(tiny.leading_hit_blocks(kept[2:]), 0)
        self.assertEqual(reply.hit_blocks, 2)

    def test_heartbeat_extends_and_expiry_refuses(self) -> None:
        hashes = block_hashes(_tokens(32), self.identity)
        self.index.insert_prefix(hashes)
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        self.clock.advance_s(20)
        extended = self.index.heartbeat(reply.pin_id)
        self.assertIsNotNone(extended)
        assert extended is not None
        self.clock.advance_s(10)
        self.assertEqual(self.index.authorize_pull(reply.pin_id), "ok")
        self.clock.advance_s(20)
        self.assertEqual(
            self.index.authorize_pull(reply.pin_id),
            MissReason.LEASE_EXPIRED.value,
        )

    def test_expired_pin_cannot_be_recycled_until_released(self) -> None:
        hashes = block_hashes(_tokens(32), self.identity)
        self.index.insert_prefix(hashes)
        reply = self.index.lookup_and_pin(
            LookupAndPinRequest(hashes, self.identity)
        )
        other = block_hashes(_tokens(32, seed=50), self.identity)
        with self.assertRaises(RuntimeError):
            self.index.reuse_ids(reply.remote_block_ids[0], other)
        self.clock.advance_s(30)
        self.assertEqual(self.index.release_expired(), 1)
        self.index.reuse_ids(reply.remote_block_ids[0], other)
        self.assertFalse(self.index.blind_read_intact(reply.pin_id))
        self.assertEqual(
            self.index.authorize_pull(reply.pin_id),
            MissReason.LEASE_EXPIRED.value,
        )


class TestFlows(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = make_cluster(num_blocks=256, lease_s=30.0)
        self.tokens = _tokens(256)

    def test_stale_index_does_not_prefill_on_a(self) -> None:
        prepare(self.cluster, "stale", self.tokens)
        owner = self.cluster.router.owner(self.cluster.A.hashes_for(self.tokens))
        self.assertEqual(owner, "engine-A")
        cached = self.cluster.A.index.leading_hit_blocks(
            self.cluster.A.hashes_for(self.tokens)
        )
        self.assertEqual(cached, 0)
        stock = run_stock(self.cluster, "stale", self.tokens)
        prepare(self.cluster, "stale", self.tokens)
        proposed = run_proposed(self.cluster, "stale", self.tokens)
        self.assertEqual(stock.a_compute_tokens, 256)
        self.assertEqual(stock.http_requests, 3)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.b_compute_tokens, 256)
        self.assertEqual(proposed.miss_reason, MissReason.EVICTED.value)
        self.assertEqual(proposed.pulled_tokens, 0)
        self.assertEqual(proposed.http_requests, 1)
        self.assertEqual(proposed.sidechannel_rpcs, 1)

    def test_full_hit_skips_a_and_pulls(self) -> None:
        prepare(self.cluster, "hit", self.tokens)
        stock = run_stock(self.cluster, "hit", self.tokens)
        prepare(self.cluster, "hit", self.tokens)
        proposed = run_proposed(self.cluster, "hit", self.tokens)
        self.assertEqual(stock.a_compute_tokens, 16)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.pulled_tokens, 256)
        self.assertEqual(proposed.b_compute_tokens, 16)
        self.assertEqual(proposed.external_tokens, 256)

    def test_partial_pull_computes_the_suffix_on_b(self) -> None:
        prepare(self.cluster, "partial", self.tokens)
        stock = run_stock(self.cluster, "partial", self.tokens)
        prepare(self.cluster, "partial", self.tokens)
        proposed = run_proposed(self.cluster, "partial", self.tokens)
        self.assertEqual(stock.a_compute_tokens, 128)
        self.assertEqual(stock.external_tokens, 256)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.pulled_tokens, 128)
        self.assertEqual(proposed.b_compute_tokens, 128)
        self.assertEqual(proposed.external_tokens, 128)
        self.assertEqual(proposed.miss_reason, MissReason.NEVER_CACHED.value)
        self.assertEqual(proposed.hit_kind, "partial")

    def test_identity_mismatch_does_not_pull(self) -> None:
        prepare(self.cluster, "identity", self.tokens)
        proposed = run_proposed(self.cluster, "identity", self.tokens)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.pulled_tokens, 0)
        self.assertEqual(proposed.b_compute_tokens, 256)
        self.assertEqual(proposed.miss_reason, MissReason.IDENTITY_MISMATCH.value)
        self.assertEqual(sum(block.ref_cnt for block in self.cluster.A.index.blocks), 0)

    def test_lease_expiry_refuses_instead_of_stale_read(self) -> None:
        prepare(self.cluster, "lease", self.tokens)
        stock = run_stock(self.cluster, "lease", self.tokens)
        prepare(self.cluster, "lease", self.tokens)
        proposed = run_proposed(self.cluster, "lease", self.tokens)
        self.assertEqual(stock.a_compute_tokens, 0)
        self.assertEqual(stock.stale_reads, 1)
        self.assertEqual(stock.lease_refuses, 0)
        self.assertEqual(stock.pulled_tokens, 256)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.lease_refuses, 1)
        self.assertEqual(proposed.stale_reads, 0)
        self.assertEqual(proposed.pulled_tokens, 0)
        self.assertEqual(proposed.b_compute_tokens, 256)

    def test_wall_clock_lease_refuse(self) -> None:
        class WallClock:
            def now(self) -> int:
                return time.time_ns()

            def advance(self, delta_ns: int) -> None:
                time.sleep(delta_ns / 1_000_000_000)

            def advance_s(self, seconds: float) -> None:
                time.sleep(seconds + 0.03)

        cluster = make_cluster(
            clock=WallClock(),  # type: ignore[arg-type]
            lease_s=0.05,
            num_blocks=128,
        )
        tokens = _tokens(64)
        prepare(cluster, "lease", tokens)
        proposed = run_proposed(cluster, "lease", tokens)
        self.assertEqual(proposed.lease_refuses, 1)
        self.assertEqual(proposed.stale_reads, 0)
        self.assertEqual(proposed.a_compute_tokens, 0)
        self.assertEqual(proposed.pulled_tokens, 0)


if __name__ == "__main__":
    unittest.main()
