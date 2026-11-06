# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers that a NixlConnector patch would call.

These functions are the behavior the patch series in PATCH_PLAN.md has to
preserve. They do not import vLLM so the CPU prototype can lock them in.
"""

from __future__ import annotations


def peer_remote_token_count(
    *,
    do_remote_prefill: bool,
    prompt_tokens: int,
    num_computed_tokens: int,
    remote_num_tokens: int | None,
    peer_lookup_resolved: bool,
    prefill_backoff: int,
) -> tuple[int, bool]:
    """External tokens ``get_num_new_matched_tokens`` should report.

    The ``do_remote_prefill`` branch in ``NixlPullConnectorScheduler``
    currently pulls the whole prompt (minus ``_prefill_backoff``). Once a
    peer pin has resolved, ``remote_num_tokens`` is the hit length and the
    scheduler computes the suffix locally in the same request. The bool is
    the connector's ``load_async`` flag: true only when there is something
    to pull.
    """

    if not do_remote_prefill:
        return 0, False
    if peer_lookup_resolved:
        hit = 0 if remote_num_tokens is None else remote_num_tokens
        count = min(hit, prompt_tokens) - num_computed_tokens
        if count < 0:
            count = 0
        return count, count > 0
    actual = prompt_tokens
    if prefill_backoff and prompt_tokens > prefill_backoff:
        actual = prompt_tokens - prefill_backoff
    count = actual - num_computed_tokens
    if count < 0:
        count = 0
    return count, count > 0


def decline_expired_lease(
    lease_deadline_unix_ns: int | None,
    now_unix_ns: int,
    safety_margin_ns: int = 0,
) -> bool:
    """Whether B must refuse the NIXL read.

    ``None`` keeps today's behavior for a producer that did not export a
    deadline (the prefill path only sets ``remote_blocks_expiry_time`` for
    bidirectional D blocks). The peer pin always sets the unix deadline.
    A read is refused when ``now + safety_margin`` has reached it.
    """

    if lease_deadline_unix_ns is None:
        return False
    return now_unix_ns + safety_margin_ns >= lease_deadline_unix_ns
