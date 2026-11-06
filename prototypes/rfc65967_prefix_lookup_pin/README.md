# RFC 65967 prototype

CPU-only model of compute-free prefix KV lookup-and-pin.
Design: [DESIGN.md](DESIGN.md). Production series: [PATCH_PLAN.md](PATCH_PLAN.md).
Numbers: [results/bench.md](results/bench.md).

Not wired into `NixlConnector`. No GPU and no PyTorch in this environment,
so the production files are unchanged.

```bash
python3 -m unittest prototypes.rfc65967_prefix_lookup_pin.test_lookup_pin -v
python3 -m prototypes.rfc65967_prefix_lookup_pin.bench
```

The bench writes `results/bench.json` and `results/bench.md`.
`a_compute_tokens` is exact. `b_ttft_control_us` is localhost TCP of the
JSON body, not GPU time-to-first-token. Lookup microseconds are an
in-process hash walk plus an O(1) pin.

[RFC_COMMENT_DRAFT.md](RFC_COMMENT_DRAFT.md) is a comment for issue 65967.
Do not post it from this tree.
