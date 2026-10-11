"""Shared closed-output instruction for industrial VLM training and inference."""

from __future__ import annotations

import json

VLM_FINDINGS_OUTPUT_CONTRACT = "industrial-vlm-findings-v1"
VLM_FINDINGS_PROMPT_CONTRACT = "industrial-vlm-json-prompt-v2"
_CONTRACT_MARKER = f"输出合同：{VLM_FINDINGS_OUTPUT_CONTRACT}"


class VlmOutputInstructionError(ValueError):
    """The VLM output instruction or controlled vocabulary is unsafe."""


def build_vlm_findings_instruction(
    instruction: str,
    *,
    allowed_labels: tuple[str, ...] = (),
) -> str:
    """Build one idempotent instruction without invalid JSON placeholders."""

    base = instruction.strip()
    if not base or len(base) > 8_000:
        raise VlmOutputInstructionError("vlm_base_instruction_is_invalid")
    if _CONTRACT_MARKER in base:
        return base
    labels = tuple(dict.fromkeys(label.strip() for label in allowed_labels))
    if any(
        not label
        or len(label) > 128
        or any(character in label for character in ("\n", "\r", "\0"))
        for label in labels
    ):
        raise VlmOutputInstructionError("vlm_controlled_label_is_invalid")
    label_rule = (
        "label 必须精确取自受控标签集合："
        + json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        + "。"
        if labels
        else "label 必须使用任务注册的受控标签，不得自行创造标签。"
    )
    return (
        f"{base}\n"
        f"{_CONTRACT_MARKER}；提示合同：{VLM_FINDINGS_PROMPT_CONTRACT}。"
        "只能返回一个单行 JSON 对象，不得返回 Markdown、解释或代码块。"
        "顶层只能有 findings 字段，findings 必须是数组。"
        "每个发现只能有 label 和 region；region 只能有 x、y、width、height。"
        "四个坐标必须是 JSON 十进制数字，不得使用文字、区间或单位；"
        "x、y 在 [0,1) 内，width、height 在 (0,1] 内，且区域不得越过图像边界。"
        f"{label_rule}"
        '没有可靠可见故障时只输出 {"findings":[]}。'
    )
