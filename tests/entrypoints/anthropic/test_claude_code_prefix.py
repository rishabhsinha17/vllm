# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-to-turn prefix fixture for Claude Code traffic on ``/v1/messages``.

Claude Code resends the whole conversation on every request and relies on
prefix caching, so the conversation rendered for request ``k`` must be reused
verbatim by request ``k+1`` and the new prefill must be exactly the appended
turn. This fixture replays a Claude Code shaped conversation (the request
shape captured in RFC #58647, with synthetic text) through
``/v1/messages/render``, which uses the same conversion and renderer as the
``/v1/messages`` server, and checks that invariant for one tokenizer per
template family.

Families that fail today are marked ``xfail(strict=True)`` with the reason, so
the file documents the defect without blocking CI (RFC #58647 item 6). A fix
that makes a family pass must drop its mark.

Assistant turns carry text and ``tool_use`` blocks only. Whether a template
keeps earlier ``thinking`` blocks is a separate property (RFC #58647 item 8)
and would mask the inline system handling this fixture isolates.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.renderers import renderer_from_config
from vllm.renderers.online_renderer import OnlineRenderer

ITEM_1 = "RFC #58647 item 1: "


@dataclass(frozen=True)
class Family:
    model: str
    revision: str
    tool_parser: str
    tokenizer_mode: str = "auto"
    model_type: str = "any"


FAMILIES = [
    pytest.param(
        Family(
            model="Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            tool_parser="hermes",
        ),
        id="qwen3",
    ),
    pytest.param(
        Family(
            model="Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            tool_parser="qwen3_xml",
        ),
        id="qwen3.5",
        marks=pytest.mark.xfail(
            strict=True,
            reason=ITEM_1 + "the template rejects non-leading system messages, "
            "so every inline system message is hoisted into the leading block",
        ),
    ),
    pytest.param(
        Family(
            model="zai-org/GLM-5.3-Flash",
            revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
            tool_parser="glm47",
        ),
        id="glm",
    ),
    pytest.param(
        Family(
            model="openai/gpt-oss-20b",
            revision="6cee5e81ee83917806bbde320786a8fb61efebee",
            tool_parser="openai",
            model_type="gpt_oss",
        ),
        id="gpt-oss",
        marks=pytest.mark.xfail(
            strict=True,
            reason=ITEM_1 + "the merge probe renders the template without "
            "strftime_now and reads the UndefinedError as a system-first guard, "
            "so the inline system messages are hoisted into the developer message",
        ),
    ),
    pytest.param(
        Family(
            model="mistralai/Ministral-3-3B-Instruct-2512",
            revision="b35d4dfe56c142746f54dbd64f579faab2744308",
            tool_parser="mistral",
            tokenizer_mode="mistral",
        ),
        id="mistral",
        marks=pytest.mark.xfail(
            strict=True,
            reason=ITEM_1 + "no Jinja template, so the converter hoists; "
            "mistral_common also folds every system message into the leading "
            "[SYSTEM_PROMPT] block",
        ),
    ),
    pytest.param(
        Family(
            model="deepseek-ai/DeepSeek-V4-Flash",
            revision="60d8d70770c6776ff598c94bb586a859a38244f1",
            tool_parser="deepseek_v4",
            tokenizer_mode="deepseek_v4",
        ),
        id="deepseek-v4",
        marks=pytest.mark.xfail(
            strict=True,
            reason=ITEM_1 + "no Jinja template, so the converter hoists although "
            "the encoder renders system messages in place",
        ),
    ),
]


@dataclass
class MockHFConfig:
    model_type: str


@dataclass
class MockModelConfig:
    model: str
    tokenizer: str
    tokenizer_mode: str
    revision: str
    tokenizer_revision: str
    hf_config: MockHFConfig
    task = "generate"
    runner_type = "generate"
    trust_remote_code = False
    code_revision = None
    max_model_len = 262144
    multimodal_config = MultiModalConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockSchedulerConfig:
    max_num_seqs: int = 128


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig = field(default_factory=MockParallelConfig)
    scheduler_config: MockSchedulerConfig = field(default_factory=MockSchedulerConfig)


def _build_serving_render(family: Family) -> ServingRender:
    """Build the render handler the way a Claude Code deployment is served.

    Only the tokenizer is loaded. ``--enable-auto-tool-choice`` and
    ``--tool-call-parser`` are set because Claude Code sends tools on every
    request.
    """
    model_config = MockModelConfig(
        model=family.model,
        tokenizer=family.model,
        tokenizer_mode=family.tokenizer_mode,
        revision=family.revision,
        tokenizer_revision=family.revision,
        hf_config=MockHFConfig(model_type=family.model_type),
    )
    renderer = renderer_from_config(MockVllmConfig(model_config))
    online_renderer = OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser=family.tool_parser,
    )
    models = OpenAIModelRegistry(
        model_config,
        [BaseModelPath(name=family.model, model_path=family.model)],
    )
    return ServingRender(models, online_renderer)


# Request shape from RFC #58647 ("Observed request shape"): a system list whose
# first block is the billing header, cache_control on the blocks, an inline
# system message after every user turn, tool results with is_error and a
# trailing reminder text block, adaptive thinking and an effort.
SYSTEM = [
    {
        "type": "text",
        "text": "x-anthropic-billing-header: cc_version=2.1.281.f3a; "
        "cc_entrypoint=cli; cch=0d1a4;",
    },
    {
        "type": "text",
        "text": "You are a coding agent working in the repository FIXTURE-SYS-A. "
        "Follow its conventions, keep edits minimal and run the tests before "
        "you finish. " * 3,
        "cache_control": {"type": "ephemeral"},
    },
    {
        "type": "text",
        "text": "Environment FIXTURE-SYS-B: cwd /repo, platform linux, shell bash.",
        "cache_control": {"type": "ephemeral"},
    },
]

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path."},
                "offset": {"type": "number"},
                "limit": {"type": "number"},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "Edit",
        "description": "Replace an exact string in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
    {
        "name": "Bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number"},
                "run_in_background": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]

TASKS = [
    "Find out why test_cache fails.",
    "Fix it without changing the public API.",
    "Add a regression test and run the suite.",
]


@dataclass(frozen=True)
class Step:
    """Messages Claude Code appends before one request, and their markers."""

    messages: list[dict[str, Any]]
    markers: tuple[str, ...]


def _text(text: str, **extra: Any) -> dict[str, Any]:
    return {"type": "text", "text": text, **extra}


def _reminder(tag: str) -> dict[str, Any]:
    return _text(f"<system-reminder>\nNote FIXTURE-REM-{tag}.\n</system-reminder>")


def _tool_use(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}


def _tool_result(call_id: str, content: Any, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
        "is_error": is_error,
    }


def _budget(budget: int) -> tuple[dict[str, Any], str]:
    text = f"<total_tokens>{budget} tokens left</total_tokens>"
    return {"role": "system", "content": text}, f"{budget} tokens left"


def _claude_code_steps() -> list[Step]:
    """Three user turns, each a two-step tool loop, in Claude Code's order."""
    steps: list[Step] = []
    budget = 200000
    final: list[dict[str, Any]] = []
    final_markers: tuple[str, ...] = ()
    for t, task in enumerate(TASKS, start=1):
        budget -= 1111 * t
        inline, inline_marker = _budget(budget)
        user_turn = {
            "role": "user",
            "content": [
                _reminder(f"{t}-0"),
                _text(f"FIXTURE-USER-{t}: {task}", cache_control={"type": "ephemeral"}),
            ],
        }
        steps.append(
            Step(
                [*final, user_turn, inline],
                (
                    *final_markers,
                    f"FIXTURE-REM-{t}-0",
                    f"FIXTURE-USER-{t}",
                    inline_marker,
                ),
            )
        )

        budget -= 2222
        inline, inline_marker = _budget(budget)
        read_a = f"/repo/src/fixture_{t}_a.py"
        steps.append(
            Step(
                [
                    {
                        "role": "assistant",
                        "content": [
                            _text(f"FIXTURE-ASST-{t}-1: I will read the module."),
                            _tool_use(f"toolu_01{t}A", "Read", file_path=read_a),
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            _tool_result(
                                f"toolu_01{t}A",
                                f"def f_{t}():\n    return FIXTURE-RES-{t}-1\n",
                                is_error=False,
                            ),
                            _reminder(f"{t}-1"),
                        ],
                    },
                    inline,
                ],
                (
                    f"FIXTURE-ASST-{t}-1",
                    read_a,
                    f"FIXTURE-RES-{t}-1",
                    f"FIXTURE-REM-{t}-1",
                    inline_marker,
                ),
            )
        )

        budget -= 3333
        inline, inline_marker = _budget(budget)
        test_b = f"tests/test_fixture_{t}_b.py"
        test_c = f"/repo/tests/test_fixture_{t}_c.py"
        steps.append(
            Step(
                [
                    {
                        "role": "assistant",
                        "content": [
                            _tool_use(
                                f"toolu_01{t}B", "Bash", command=f"pytest {test_b}"
                            ),
                            _tool_use(f"toolu_01{t}C", "Read", file_path=test_c),
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            _tool_result(
                                f"toolu_01{t}B",
                                [_text(f"1 failed FIXTURE-RES-{t}-2b")],
                                is_error=True,
                            ),
                            _tool_result(
                                f"toolu_01{t}C",
                                f"import pytest  # FIXTURE-RES-{t}-2c\n",
                                is_error=False,
                            ),
                            _reminder(f"{t}-2"),
                        ],
                    },
                    inline,
                ],
                (
                    test_b,
                    test_c,
                    f"FIXTURE-RES-{t}-2b",
                    f"FIXTURE-RES-{t}-2c",
                    f"FIXTURE-REM-{t}-2",
                    inline_marker,
                ),
            )
        )

        final = [
            {"role": "assistant", "content": f"FIXTURE-ASST-{t}-3: Done with step {t}."}
        ]
        final_markers = (f"FIXTURE-ASST-{t}-3",)
    return steps


def _request(model: str, messages: list[dict[str, Any]]) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest.model_validate(
        {
            "model": model,
            "max_tokens": 32000,
            "system": SYSTEM,
            "messages": messages,
            "tools": TOOLS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "metadata": {"user_id": "fixture-user"},
        }
    )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.asyncio
@pytest.mark.parametrize("family", FAMILIES)
async def test_claude_code_turns_extend_rendered_prefix(family: Family):
    serving = _build_serving_render(family)
    tokenizer = serving.online_renderer.renderer.tokenizer
    assert tokenizer is not None
    steps = _claude_code_steps()

    renders: list[list[int]] = []
    messages: list[dict[str, Any]] = []
    for step in steps:
        messages.extend(step.messages)
        result = await serving.render_messages_request(_request(family.model, messages))
        assert not isinstance(result, ErrorResponse), result.error.message
        renders.append(list(result.token_ids))

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    seen: list[str] = ["FIXTURE-SYS-A", "FIXTURE-SYS-B", *steps[0].markers]
    for k, (prev, cur) in enumerate(zip(renders, renders[1:]), start=1):
        step = steps[k]
        kept = _common_prefix_len(prev, cur)
        # Only the generation prompt may be rendered differently once the
        # assistant turn that replaces it is in the history.
        assert not any(m in decode(prev[kept:]) for m in seen), (
            f"request {k + 1} reuses {kept} of request {k}'s {len(prev)} rendered "
            f"tokens; from there request {k} had {decode(prev[kept : kept + 32])!r} "
            f"and request {k + 1} has {decode(cur[kept : kept + 32])!r}"
        )
        prefill = decode(cur[kept:])
        assert all(m in prefill for m in step.markers), (
            f"request {k + 1}'s new prefill lacks part of the appended turn: "
            f"{[m for m in step.markers if m not in prefill]} not in {prefill!r}"
        )
        assert not any(m in prefill for m in seen), (
            f"request {k + 1}'s new prefill re-renders earlier content: "
            f"{[m for m in seen if m in prefill]}"
        )
        seen.extend(step.markers)
