"""Quick test for tools (function calling) support."""
import asyncio
import os
from llm_provider import (
    LLMProviderFactory,
    OpenAIConfig,
    ToolFunction,
    GenerationRequest,
    Message,
    MessageRole,
)


def test_tool_models():
    """Test that tool models can be created."""
    tool = ToolFunction(
        name="get_weather",
        description="Get the current weather in a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )
    print(f"✅ ToolFunction model created: {tool.name}")
    return tool


async def test_openai_tools():
    """Test OpenAI provider with tools."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI tools test")
        return

    factory = LLMProviderFactory()
    provider = factory.create_openai(
        OpenAIConfig(api_key=api_key, model="gpt-4o-mini")
    )

    weather_tool = ToolFunction(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )

    request = GenerationRequest(
        messages=[
            Message(
                role=MessageRole.USER, content="What's the weather in San Francisco?"
            )
        ],
        tools=[weather_tool],
        tool_choice="auto",
        temperature=0.2,
    )

    try:
        response = await provider.generate(request)
        print(f"✅ OpenAI response received: {response.finish_reason}")

        if response.tool_calls:
            print(f"✅ Tool calls detected: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"  - Tool: {tc.name}, Args: {tc.arguments}")
        else:
            print(f"⚠️  No tool calls (content: {response.content[:50]}...)")

    except Exception as e:
        print(f"❌ OpenAI tools test failed: {e}")


async def main():
    print("🧰 Testing Tools / Function Calling Support\n")

    # Test 1: Tool models
    test_tool_models()
    print()

    # Test 2: OpenAI tools
    await test_openai_tools()
    print()

    print("✨ Tools support is ready!")


if __name__ == "__main__":
    asyncio.run(main())
