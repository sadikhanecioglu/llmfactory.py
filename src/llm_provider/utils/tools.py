"""Tool execution helpers for function-calling capable models.

Provides a simple, provider-agnostic tool execution loop.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional, Union, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..base_provider import BaseLLMProvider

from ..settings import (
    GenerationRequest,
    GenerationResponse,
    Message,
    MessageRole,
    ToolFunction,
)

ExecuteToolFunc = Callable[[str, Dict[str, Any]], Union[str, Dict[str, Any]]]


def _coerce_arguments(args: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except Exception:
        # Fallback: provide raw string in a field
        return {"_raw": args}


async def run_with_tools_async(
    provider: "BaseLLMProvider",
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    tools: List[ToolFunction],
    execute_tool: ExecuteToolFunc,
    max_iterations: int = 3,
    temperature: Optional[float] = None,
) -> GenerationResponse:
    """Run a tool-enabled conversation loop until completion (async)."""
    if messages is None:
        if not prompt:
            raise ValueError("Either messages or prompt must be provided")
        messages = [Message(role=MessageRole.USER, content=prompt)]

    history = list(messages)

    for _ in range(max_iterations):
        req = GenerationRequest(
            messages=history,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        resp = await provider.generate(req)

        if resp.tool_calls:
            # First, add the assistant's message with tool calls to history
            history.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=resp.content or "",
                )
            )
            # Then add tool results
            for tc in resp.tool_calls:
                args = _coerce_arguments(tc.arguments)
                result = execute_tool(tc.name, args)
                if not isinstance(result, str):
                    try:
                        result = json.dumps(result)
                    except Exception:
                        result = str(result)
                history.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result,
                        name=tc.name,
                        tool_call_id=tc.id or "",
                    )
                )
            continue
        else:
            return resp

    # Max iterations reached; ask model to finalize without further tools
    req = GenerationRequest(messages=history, tool_choice="none", tools=tools)
    final_resp = await provider.generate(req)
    return final_resp


def run_with_tools(
    provider: "BaseLLMProvider",
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    tools: List[ToolFunction],
    execute_tool: ExecuteToolFunc,
    max_iterations: int = 3,
    temperature: Optional[float] = None,
) -> GenerationResponse:
    """Synchronous wrapper around run_with_tools_async for convenience."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In running loop; create a task and wait
        return asyncio.run_coroutine_threadsafe(
            run_with_tools_async(
                provider,
                prompt=prompt,
                messages=messages,
                tools=tools,
                execute_tool=execute_tool,
                max_iterations=max_iterations,
                temperature=temperature,
            ),
            loop,
        ).result()
    else:
        return asyncio.run(
            run_with_tools_async(
                provider,
                prompt=prompt,
                messages=messages,
                tools=tools,
                execute_tool=execute_tool,
                max_iterations=max_iterations,
                temperature=temperature,
            )
        )
