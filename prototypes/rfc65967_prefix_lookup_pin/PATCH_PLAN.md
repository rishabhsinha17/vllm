# Patch plan: NixlConnector lookup-and-pin

Four commits. None of them are applied in this tree. The behavior they
have to preserve is `nixl_hooks.py`, locked by
`test_lookup_pin.py`. Apply this on a machine with the vLLM venv and
extend `tests/v1/kv_connector/unit/test_nixl_connector.py` rather than
treating the CPU mock as a GPU test.

Do not change today's `do_remote_prefill` result unless
`peer_lookup_resolved` is set. Do not decline a read whose
`lease_deadline_unix_ns` is absent.

## Commit 1 — wire types

File: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`

* `LOOKUP_PIN_MSG = b"lookup_pin"`.
* `MissReason`, `HitKind`, `KvIdentity`, `LookupAndPinRequest`, `PinReply`
  as in `protocol.py`. msgspec structs, not dataclasses, so they match
  `NixlHandshakePayload`.
* `RemoteMeta` gains `lease_deadline_unix_ns: int | None = None` and
  `lease_extension_ns: int | None = None`. Keep `blocks_expiry_time` for
  the perf_counter turn-2 path.
* `NixlConnectorMetadata._add_new_req` copies the new fields from
  `kv_transfer_params`.
* `NIXL_CONNECTOR_VERSION` 12 to 13. The new message is an incompatible
  side-channel verb; old peers should fail the compatibility hash.

`kv_identity.compatibility_hash` is `compute_nixl_compatibility_hash`
plus the request `cache_salt` and `block_size`. Do not fold the model
identity into the block hash. The block hash already carries `cache_salt`
as an extra key; the explicit compare catches a caller who hashed with
the right salt and the wrong model.

## Commit 2 — lookup, pin, side channel

Files:

* `vllm/v1/core/kv_cache_manager.py`
* `vllm/v1/core/block_pool.py` (tombstone only if the event path is not
  enough to query)
* `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`
* `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py`

`KVCacheManager.lookup_blocks_by_hash(block_hashes) -> (blocks, miss)`:

* No `Request`. No call to `get_computed_blocks`.
* Walk `coordinator.find_longest_cache_hit` or, for a single full-attention
  group, `BlockPool.get_cached_block` in order.
* Return the blocks and the stop reason. Do not `touch` inside the
  manager if the connector is the one that decides to pin; do `touch`
  in the connector while still holding the pool lock so a miss cannot
  be evicted before the lease is recorded. One lock scope is simpler:
  the connector calls a manager method that walks and touches, and the
  connector records the lease before releasing the lock.

Tombstones: on `_emit_block_removed_events`, also insert the external
hash into an `OrderedDict` capped at a few thousand entries. Lookup of a
missing hash that is still in the map returns `evicted`. Otherwise
`never_cached`.

Lease table on `NixlBaseConnectorScheduler`, beside `_reqs_need_send`:

```text
pin_id -> (block ids per group, deadline_unix_ns, extension_ns)
```

Sweep it where expired `_reqs_need_send` entries are already dropped.
`touch` is the pin (`BlockPool.touch`). Sweep and read-completion call
the same free path as `request_finished` giving up the lease (drop one
refcount per pin, not `evict_blocks`, so a prefix that is still cached
can be pinned again).

Side channel: extend `_nixl_handshake_listener`. Today it decodes
`(GET_META_MSG, pp_rank, tp_rank)` and answers with handshake bytes plus
a perf_counter sample. Add a branch for `LOOKUP_PIN_MSG` that:

1. Decodes `LookupAndPinRequest`.
2. Takes `block_pool_lock`.
3. Runs lookup-and-pin.
4. Releases the lock.
5. Sends `PinReply`.

`schedule()` takes the same lock only around pool mutations, and never
around an outbound lookup. `EngineCore.step` does not need a new poll
point if the listener thread can run during `execute_model`.

`NixlPullConnectorScheduler.get_num_new_matched_tokens`: if
`params["peer_lookup"]` and not `peer_lookup_resolved`, call
`_lookup_and_pin_remote(request)` **before** taking the pool lock (the
caller in `schedule()` must drop it first; see DESIGN.md). On success
write `remote_block_ids`, `remote_num_tokens=hit_tokens`,
`remote_request_id=pin_id`, `lease_deadline_unix_ns`, and
`peer_lookup_resolved=True`. On mismatch or a total miss, set
`remote_num_tokens=0` and do not set block ids.

Timeout: if A does not answer, resolve as a miss and compute locally.
Do not reuse router block ids after a failed lookup.

## Commit 3 — partial remote prefix

File: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py`

`get_num_new_matched_tokens`, `do_remote_prefill` branch. Current code
(`pull_scheduler.py`):

```python
if params is not None and params.get("do_remote_prefill"):
    token_ids = request.prompt_token_ids or []
    actual = self._get_remote_prefill_token_count(len(token_ids))
    count = actual - num_computed_tokens
    if count > 0:
        return count, True
```

New branch, in front of that, matching `peer_remote_token_count`:

```python
if params is not None and params.get("do_remote_prefill"):
    if params.get("peer_lookup_resolved"):
        hit = int(params.get("remote_num_tokens") or 0)
        count = min(hit, len(request.prompt_token_ids or [])) - num_computed_tokens
        count = max(count, 0)
        return count, count > 0
    # existing full-prompt path, unchanged
```

`update_state_after_alloc` already builds `ReqMeta` from
`remote_block_ids` and only waits for remote KV when
`num_external_tokens > 0`. A partial hit passes the pinned ids, not the
whole prompt. The local suffix is the ordinary prefill in the same
schedule step. Keep `_prefill_backoff` on the legacy full-prompt path.
For a peer pin, the hit length is already block-aligned; B's local
schedule recomputes the last block when the hit covers the prompt, which
is `scheduled_prefill_tokens` in `protocol.py`.

No worker change for the slice itself when `remote.num_tokens` is the
hit length. `NixlPullConnectorWorker._read_blocks_for_req` already
derives `num_remote_blocks` from `meta.remote.num_tokens`.

## Commit 4 — refuse an expired lease

File: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`

`_is_turn2_read_expired` only runs when bidirectional transfer is on,
and only against `remote_blocks_expiry_time` (a remote perf_counter).
Generalize:

```python
def _lease_expired(self, meta: ReqMeta) -> bool:
    remote = meta.remote
    if remote is None:
        return False
    deadline = remote.lease_deadline_unix_ns
    if deadline is None:
        # Legacy producer. Turn-2 keeps the old perf_counter check.
        return (
            self._bidirectional_kv_xfer_enabled
            and self._is_turn2_read_expired(meta)
        )
    return decline_expired_lease(deadline, time.time_ns(), safety_margin_ns)
```

Call it at the top of `_read_blocks_for_req` for every pull, not only
turn-2. On decline: do not build a NIXL read, record
`nixl_stale_lease_declines`, and `_handle_failed_transfer(...,
record_failed_transfer=False)` so `kv_load_failure_policy` recomputes.
Log the pin id and the deadline. Do not log the block contents.

`decline_expired_lease` is the predicate in `nixl_hooks.py`. Tests to
add beside the existing NIXL unit tests:

* Full hit: pin refs go up, prefill tokens scheduled on A stay 0.
* Evicted hash: `miss_reason=evicted`, no `touch`.
* Never cached: `miss_reason=never_cached`.
* Identity mismatch: no block ids, refs unchanged.
* Partial: `get_num_new_matched_tokens` returns the hit, not the prompt.
* Deadline in the past: worker does not call `_read_blocks`.
* Deadline absent: today's turn-2 check only, prefill pulls still read.

## Router

Out of scope for these commits. The one request B receives is the
sketch in DESIGN.md. The router keeps its KV index for choosing B; it
stops sending the probe to A.

## Tests that need a GPU

Not this series. After it lands, a 1P1D run (two processes, shared
prefix, block size 16) should confirm:

* Stale index: A logs no prefill for the peer request; B's prompt
  tokens equal the prefix.
* Fresh hit: A logs no prefill; B's external tokens equal the prefix
  and B's scheduled prefill is one block.
* Lease shorter than the injected delay: `nixl_stale_lease_declines`
  increments and the decode logits are not NaN.

The CPU harness in `bench.py` is the stand-in until that machine exists.
It does not measure NIXL bandwidth or GPU TTFT.
