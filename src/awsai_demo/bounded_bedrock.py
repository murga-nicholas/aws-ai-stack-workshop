"""Normalize counted Bedrock requests before the SDK dispatches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands.models.bedrock import BedrockModel

if TYPE_CHECKING:
    from strands.types.content import Messages, SystemContentBlock
    from strands.types.tools import ToolChoice, ToolSpec


class BoundedBedrockModel(BedrockModel):
    """Avoid Strands' unreserved validation-repair inference retry."""

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
        dynamic_trailing_blocks: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply the SDK's idempotent repair before token counting."""
        request = super().format_request(
            messages,
            tool_specs,
            system_prompt_content,
            tool_choice,
            dynamic_trailing_blocks,
            **kwargs,
        )
        # The SDK retry sends again only if this transformation changes
        # the request. Applying it here makes that branch a no-op.
        request["messages"] = self._separate_tool_result_turns(
            request["messages"]
        )
        return request
