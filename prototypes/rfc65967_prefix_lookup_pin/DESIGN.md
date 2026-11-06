# Compute-free prefix KV lookup-and-pin

Prototype for [vLLM RFC #65967](https://github.com/vllm-project/vllm/issues/65967).
This note answers the three maintainer questions and sketches the API.
The executable spec is the CPU-only package next to this file. It is not
wired into `NixlConnector`: this environment has no GPU and no PyTorch, so
a production edit would be untested. `PATCH_PLAN.md` is the reviewable
series for that wiring.

## Answers

**1. Scheduler-side primitive, connector-owned lease.**
The hash walk belongs on `KVCacheManager`, which already owns
`BlockHashToBlockMap`. The lease belongs on `NixlPullConnectorScheduler`,
next to `_reqs_need_send`, reusing `kv_lease_duration`. Do not build a
second hash index inside the connector, and do not enter through
`get_computed_blocks`. That function exists to feed a request that will
run, and it deliberately drops the last token so the model can emit
logits. Lookup-and-pin must not allocate a `Request` and must not schedule
a prefill.

**2. Connector-to-connector is the primary path.**
B calls A on the existing NIXL side channel at pull time
(`LOOKUP_PIN_MSG` beside `GET_META_MSG`). That checks the cache A still
has, and it skips A's API server, tokenizer, and scheduler admission.
An engine RPC with the same struct can come later for a router that wants
the hit length before it picks B. That call is optional: it adds a round
trip and the lease has to survive until B pulls. It should not be the
only path.

**3. Partial remote prefix is open for contribution.**
`NixlPullConnectorScheduler.get_num_new_matched_tokens` treats
`do_remote_prefill` as the whole prompt (minus `_prefill_backoff`). The
scheduler already computes a local suffix after
`num_external_computed_tokens`, and the bidirectional path already passes
`remote_num_tokens`. The missing piece is to feed the pin's hit length
into that path. This is not a new transfer engine. It is the peer-prefix
argument on a pull that already exists for the D-to-P turn-2 case.

## What is wrong with the 3-request flow

Stock peer reuse is:

1. Router to A: prefix only, `do_remote_decode`, `max_tokens=1`. A leases
   the blocks (`kv_lease_duration`, 30s) and returns their ids.
2. Router to B: `do_remote_prefill`, `max_tokens=1`. B pulls every prompt
   block.
3. Router to B: the real request, which hits B's local cache.

Two gaps the router cannot close:

* A's index in the router is eventually consistent. If A has evicted the
  prefix and the `BlockRemoved` event is still in flight, step 1 is a
  normal prefill on the worker the router was trying to unload. Nothing
  in the request says "cache only, otherwise refuse". `cached_tokens`
  comes back after the work.
* The pull is all-or-nothing. `get_num_new_matched_tokens` returns the
  full prompt for `do_remote_prefill`, so B cannot take a prefix from A
  and compute the rest in that same request. That is why step 2 exists.

A full cache hit still schedules the last block on A. `get_computed_blocks`
caps the hit at `num_tokens - 1` and the scheduler floors to a block
boundary, so a block-aligned prefix recomputes `block_size` tokens just
to serve the probe.

## API

### Request on B (one HTTP call)

The router sends the real request to B. Peer fields sit in
`kv_transfer_params` until B resolves them. Hashes are full blocks in
prefix order, the same chained hash as `hash_block_tokens`. The short
tail below `block_size` is not hashed; B computes it locally.

```json
{
  "do_remote_prefill": true,
  "peer_lookup": true,
  "remote_engine_id": "engine-A",
  "remote_host": "10.0.0.8",
  "remote_port": 5567,
  "remote_block_hashes": ["<hex>", "..."],
  "kv_identity": {
    "compatibility_hash": "<compute_nixl_compatibility_hash>",
    "cache_salt": "tenant-a",
    "block_size": 16,
    "hash_algo": "sha256"
  }
}
```

`compatibility_hash` is the existing NIXL handshake hash (model, dtype,
KV heads, connector version, transfer mode). It is not inside the block
hash: two models with the same tokenizer hash the same token ids and must
still refuse. `cache_salt` is the extra key `generate_block_hash_extra_keys`
mixes into the first block. A declared salt that does not match A's
identity fails closed even if the caller hashed with A's salt.

### Side-channel RPC

`LOOKUP_PIN_MSG = b"lookup_pin"` on the handshake listener socket.
Body is msgpack of:

```python
class LookupAndPinRequest:
    block_hashes: tuple[bytes, ...]
    kv_identity: KvIdentity
    lease_duration_s: float | None  # default: kv_lease_duration
```

A's reply:

```python
class MissReason(str, Enum):
    NONE = "none"                    # every requested full block was pinned
    NEVER_CACHED = "never_cached"    # walk stopped on a hash A has not stored
    EVICTED = "evicted"              # walk stopped on a hash A has removed
    IDENTITY_MISMATCH = "identity_mismatch"  # fail closed, no block ids
    LEASE_EXPIRED = "lease_expired"  # pull-time only, not a lookup result

class HitKind(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    MISS = "miss"

class PinReply:
    hit_kind: HitKind
    hit_tokens: int                  # hit_blocks * block_size; 0 on a miss
    hit_blocks: int
    remote_block_ids: tuple[tuple[int, ...], ...]  # per transfer group
    pin_id: str                      # synthetic remote_request_id
    engine_id: str
    host: str
    port: int
    lease_deadline_unix_ns: int      # time.time_ns, 0 if nothing was pinned
    lease_extension_ns: int          # kv_lease_duration * 2/3
    heartbeat_interval_ns: int       # kv_lease_duration // 6
    miss_reason: MissReason          # why the walk stopped; NONE iff full
    kv_identity: KvIdentity          # A's identity on success
    transfer_mode: str               # "pull"
```

Read `hit_kind` before treating `miss_reason` as a total miss. A partial
hit has `hit_tokens > 0` and `miss_reason` set to why the next block
stopped (`never_cached` or `evicted`).

`lease_deadline_unix_ns` is wall time, not `time.perf_counter`.
perf_counter epochs differ across processes; the connector already rebases
`scheduler_clock` for that reason, and #66009 is the stale-read that
follows when a consumer cannot see the deadline. `lease_extension_ns` is
the heartbeat quantum so B can advance a local model of the deadline when
A accepts an HB (`max(old, now + extension)`, same as today's handler).

After B resolves the pin it rewrites `kv_transfer_params` into the shape
the pull worker already understands, plus the deadline:

```json
{
  "do_remote_prefill": true,
  "peer_lookup_resolved": true,
  "remote_block_ids": [[1, 2, 3]],
  "remote_engine_id": "engine-A",
  "remote_request_id": "pin-...",
  "remote_host": "10.0.0.8",
  "remote_port": 5567,
  "remote_num_tokens": 4096,
  "lease_deadline_unix_ns": 1730000000123456789,
  "lease_extension_ns": 20000000000,
  "hit_tokens": 4096,
  "miss_reason": "none",
  "transfer_mode": "pull"
}
```

On `identity_mismatch` or a total miss, `remote_num_tokens` is 0 and
`remote_block_ids` is omitted. B computes the prompt locally. No pull.

Bump `NIXL_CONNECTOR_VERSION` (now 12) when the message is added so an
old peer fails the handshake instead of ignoring `LOOKUP_PIN_MSG`.

### Lookup rules

On A's scheduler thread, under a block-pool lock that `schedule()` holds
only around its own pool mutations:

1. If `kv_identity` does not match this engine, return
   `IDENTITY_MISMATCH`, zero blocks, and do not touch the pool.
2. Walk `block_hashes` in order against `BlockHashToBlockMap` /
   `BlockPool.get_cached_block`. Stop at the first miss.
3. The first missing hash is `EVICTED` if it is in a bounded tombstone
   set filled from the same path as `_emit_block_removed_events`, else
   `NEVER_CACHED`. A tombstone that has aged out reports `never_cached`.
   Either way the hit length is honest and nothing is pulled past it.
4. `BlockPool.touch` every hit block (this is the pin: refcount, and the
   block leaves the free queue). Record a lease keyed by `pin_id` with
   deadline `time.time_ns() + kv_lease_duration`.
5. Return. Do not call `schedule`, do not increment prefill tokens.

Expiry sweep drops the refs, same as `_reqs_need_send` timing out.
`complete_pull` drops them early when B's read-completion notification
arrives. A heartbeat extends with `max(old, now + duration * 2/3)`.

B refuses the NIXL read when
`now_unix_ns + safety_margin_ns >= lease_deadline_unix_ns`. A missing
deadline stays on today's path so existing proxies that omit the field
do not change. The peer path always sets it. Refusing follows
`kv_load_failure_policy` (recompute locally) and increments
`nixl_stale_lease_declines`. This is the #66009 check, using unix time
so two processes do not compare perf_counter epochs. A small margin
(about 1s) covers NTP skew. It does not need the 5s margin
`_KV_BLOCKS_EXPIRY_SAFETY_MARGIN` uses to absorb a bogus perf_counter
offset on the turn-2 path.

### Partial pull

Replace the `do_remote_prefill` branch of `get_num_new_matched_tokens`:

```python
if params.get("peer_lookup_resolved"):
    hit = int(params.get("remote_num_tokens") or 0)
    count = min(hit, prompt_tokens) - num_computed_tokens
    return max(count, 0), count > 0
# unchanged: whole prompt, minus _prefill_backoff
```

`peer_remote_token_count` in `nixl_hooks.py` is that branch, tested
without vLLM. The worker already reads `meta.local_block_ids` and, when
`remote_num_tokens` is set, slices the transfer with
`_apply_prefix_caching_by_region`. Returning fewer external tokens is
what makes the suffix local. No new assert that the pull covers the
prompt.

### Where the RPC runs

`BlockPool` is single-threaded today. The handshake listener is a
background thread. Serving the lookup on that thread without a lock
races `schedule()`. Serving it only on the scheduler thread cannot run
during `future.result()` in `EngineCore.step`, so a lookup would wait
out A's current prefill. That still avoids the extra compute, but it
puts A's step latency on B's critical path, which is the case the RFC
is about.

First cut that stays correct under load:

* Add a `block_pool` lock. `schedule()` holds it only around local pool
  mutations. The lookup handler on the side-channel thread holds it only
  for the walk and `touch`.
* B sends the RPC from outside the lock. A thread that holds the lock
  must not wait on a peer lookup (that deadlocks if both peers pin each
  other). The handler itself only touches the local pool, so it does
  not recurse.
* Timeout the RPC (on the order of one decode step, not one prefill).
  On timeout B computes locally. A lost reply is a miss, not a pull of
  guessed block ids.
* Two peers that look each other up in the same step can wait on each
  other's RPC. The timeout breaks that. No request being served as a
  lookup may itself start a peer lookup.

## What this prototype measured

CPU-only mock of two workers. No GPU, no NIXL byte transfer, no weights.
See `results/bench.md`.

The number the RFC asks for is exact, because the flow actually schedules
it. On a stale index (router still names A, A has evicted, tombstone
kept) A's prefill tokens are the full prefix in the 3-request flow and
0 in the 1-request flow, at 256, 1024, 4096, and 16384 tokens. The
prefill moves to B. Aggregate tokens do not disappear on a total miss;
they stop running on the busy worker, and the stock path also pays one
extra block on B after A has recomputed everything.

A full hit pins the prefix with A at 0. Stock still schedules one block
on A for the probe. A 50% hit pulls half and computes the suffix on B;
stock prefills that suffix on A because the pull cannot be partial.

Lease: after the deadline, A sweeps the pin and a later request reuses
the block ids. Stock's warmup trusts the ids (1 stale read, 0 refuses).
The proposed pull refuses (1 refuse, 0 stale reads, A compute 0) and B
recomputes. One deterministic case at 1024 tokens, plus a unit test that
sleeps past a 50ms wall-clock lease. This is not a load-test miss rate.

In-process lookup-and-pin (no hashing, no TCP) on a stale miss returns
at the first tombstone, about 2us at 256 through 16384 tokens. A full
hit, walk plus pin, is about 23us at 256 tokens and about 0.41ms at
16384. Localhost TCP of the JSON body is a few milliseconds and is
labeled control-plane time, not GPU TTFT. No microseconds-per-token
constant was applied.

## Non-goals

* Push mode. The same pin reply can feed a push later; this cut is pull.
* Hybrid / Mamba groups. The production walk should use the full-attention
  hit the way `get_computed_blocks_for_connector` does, and refuse if a
  lagging group cannot be transferred with the KV. The mock is one group.
* Router changes. The router sends hashes and `kv_identity` on the single
  request to B. It does not call A.
