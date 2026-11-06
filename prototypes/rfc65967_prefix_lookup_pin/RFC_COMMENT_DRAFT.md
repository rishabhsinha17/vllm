# Draft comment for vLLM RFC #65967

**Do not post this.** It is a draft for the parent run. Nothing here was
submitted to GitHub.

---

CPU-only mock of the lookup-and-pin path, two workers, block size 16.
No GPU, no NIXL transfer, no model weights, so this is not a Qwen TTFT.
A's compute is the prefill the executed flow actually schedules.
"B TTFT" below is p50 localhost TCP of the JSON body over 21 iterations
(control plane only), rounded to the microsecond. In-process
lookup-and-pin (no hashing, no TCP) on a stale miss is about 2us at
every length tried, because the walk stops on the first tombstone. A
full hit is 23 / 29 / 107 / 410us at 256 / 1024 / 4096 / 16384 tokens.
No us-per-token constant was applied. Harness:
`prototypes/rfc65967_prefix_lookup_pin/` on the fork, not merged.

Stale router index (A evicted, `BlockRemoved` not applied, tombstone kept).
Stock is the 3-request probe. Proposed is one request to B, B calls
lookup-and-pin on A.

| tokens | stock A compute | proposed A compute | stock critical-path tokens | proposed critical-path tokens | stock control us | proposed control us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 0 | 272 | 256 | 1752 | 1433 |
| 1024 | 1024 | 0 | 1040 | 1024 | 2479 | 1722 |
| 4096 | 4096 | 0 | 4112 | 4096 | 5235 | 3051 |
| 16384 | 16384 | 0 | 16400 | 16384 | 15539 | 8000 |

Proposed miss reason is `evicted`, hit length 0, nothing pulled. The
prefill moves onto B (B compute = prefix). It does not vanish. Stock
also schedules one extra block on B after A has recomputed the prefix,
because a full hit still recomputes the last block for logits. That is
why stock's critical path is `prefix + 16`.

Full hit: stock still schedules 16 tokens on A (the probe). Proposed
schedules 0 on A, pulls the prefix, and B schedules 16. Partial hit at
50%: stock prefills the missing suffix on A (128 / 512 / 2048 / 8192
tokens) because `do_remote_prefill` pulls the whole prompt. Proposed
pulls the half A still has (`hit_kind=partial`, suffix reason
`never_cached`) and B computes that suffix. A stays at 0.

Lease, one deterministic case at 1024 tokens: pin, expire, recycle the
block ids. Stock warmup does not see a deadline: `stale_reads=1`,
`lease_refuses=0`, pulled 1024. The mock flags the recycled block id.
It does not copy KV bytes or produce logits. Proposed refuses:
`lease_refuses=1`, `stale_reads=0`, pulled 0, A compute 0, B recomputes
1024. A unit test also sleeps past a 50ms wall-clock lease and refuses.
This is not a load-test rate. It is the #66009 read, declined because
the pin reply carries `lease_deadline_unix_ns`.

Identity: same token hashes, different `compatibility_hash`. Proposed
returns `identity_mismatch`, 0 block ids, 0 refs taken, A compute 0,
B computes the prompt. Fail closed.

On the three questions, from the prototype rather than as a preference:

1. Put the walk on `KVCacheManager` (it owns `BlockHashToBlockMap`) and
   the lease on the NIXL scheduler next to `_reqs_need_send`. Do not
   enter through `get_computed_blocks`; that path drops the last token
   and exists to run a request.
2. B should call A on the side channel (`LOOKUP_PIN_MSG`) at pull time.
   A router RPC can return the same `PinReply`, but it is a second hop
   and the lease has to hold until B pulls.
3. Partial prefix is a missing argument on the existing pull.
   `get_num_new_matched_tokens` would return `remote_num_tokens` once
   the pin has resolved, instead of the whole prompt. The scheduler
   already computes the suffix.

Pin reply fields that mattered in the mock: `hit_kind`, `hit_tokens`,
`miss_reason` (`none | never_cached | evicted | identity_mismatch`),
`remote_block_ids`, `lease_deadline_unix_ns`, `lease_extension_ns`
(`kv_lease_duration * 2/3`). `lease_expired` is pull-time, not a lookup
result. Deadline is unix time, not `perf_counter`.
