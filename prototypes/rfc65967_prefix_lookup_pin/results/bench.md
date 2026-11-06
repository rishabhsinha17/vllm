# RFC 65967 microbench (CPU-only mock)

No GPU, no NIXL byte transfer, no model weights.
A/B compute columns are exact scheduled prefill tokens.
TTFT columns are p50 localhost TCP control-plane time in microseconds,
including JSON of the token ids, over 21 iterations. They are not GPU time-to-first-token.
The lookup column is in-process lookup-and-pin for that scenario,
with no block hashing and no TCP.

| scenario | tokens | stock A | proposed A | stock B | proposed B | stock critical | proposed critical | stock TTFT us | proposed TTFT us | lookup us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stale | 256 | 256 | 0 | 16 | 256 | 272 | 256 | 1752.1 | 1433.3 | 2.9 |
| stale | 1024 | 1024 | 0 | 16 | 1024 | 1040 | 1024 | 2478.9 | 1722.0 | 1.6 |
| stale | 4096 | 4096 | 0 | 16 | 4096 | 4112 | 4096 | 5234.8 | 3050.7 | 1.9 |
| stale | 16384 | 16384 | 0 | 16 | 16384 | 16400 | 16384 | 15539.4 | 7999.8 | 1.9 |
| hit | 256 | 16 | 0 | 16 | 16 | 32 | 16 | 1871.4 | 1396.5 | 22.5 |
| hit | 1024 | 16 | 0 | 16 | 16 | 32 | 16 | 2705.7 | 1886.9 | 29.2 |
| hit | 4096 | 16 | 0 | 16 | 16 | 32 | 16 | 4991.7 | 3786.4 | 107.0 |
| hit | 16384 | 16 | 0 | 16 | 16 | 32 | 16 | 15108.5 | 11269.3 | 409.5 |
| partial | 256 | 128 | 0 | 16 | 128 | 144 | 128 | 1596.2 | 1345.9 | 8.6 |
| partial | 1024 | 512 | 0 | 16 | 512 | 528 | 512 | 2182.9 | 1704.1 | 17.7 |
| partial | 4096 | 2048 | 0 | 16 | 2048 | 2064 | 2048 | 4580.6 | 3418.0 | 56.2 |
| partial | 16384 | 8192 | 0 | 16 | 8192 | 8208 | 8192 | 15721.2 | 10738.3 | 216.9 |

## Lease expiry at 1024 tokens

- Stock blind read after recycle: stale_reads=1, refuses=0, pulled=1024.
- Proposed: refuses=1, stale_reads=0, A compute=0, pulled=0, B recomputes 1024 tokens.

## Identity mismatch at 1024 tokens

- miss_reason=identity_mismatch, A compute=0, pulled=0, B compute=1024.

