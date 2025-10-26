import asyncio
import os
from llm_provider import (
    LLMProviderFactory,
    OpenAIConfig,
    Message, MessageRole,
    ToolFunction,
)
from llm_provider.utils import run_with_tools_async


# Define a simple calculator tool
calculator_tool = ToolFunction(
    name="calculator",
    description="Evaluate a basic arithmetic expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression, e.g. '2+2*3'"}
        },
        "required": ["expression"],
    },
)


def execute_tool(name: str, args: dict):
    if name == "calculator":
        expr = args.get("expression", "")
        try:
            # VERY limited safe eval for demo only
            allowed = {"__builtins__": None}
            return str(eval(expr, allowed, {}))
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    factory = LLMProviderFactory()
    provider = factory.create_openai(OpenAIConfig(api_key=api_key, model="gpt-4o-mini"))

    response = await run_with_tools_async(
        provider,
        prompt="What's (12 + 30) * 2? Use the calculator tool.",
        tools=[calculator_tool],
        execute_tool=execute_tool,
        max_iterations=2,
        temperature=0.2,
    )

    print("Assistant:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
