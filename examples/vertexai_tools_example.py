"""
VertexAI Gemini ile Tools / Function Calling Örneği

Google Cloud Vertex AI Gemini modellerinin function calling özelliğini gösterir.
"""

import asyncio
import os
from llm_provider import (
    LLMProviderFactory,
    VertexAIConfig,
    ToolFunction,
    Message,
    MessageRole,
)
from llm_provider.utils import run_with_tools_async


# Weather tool tanımı
weather_tool = ToolFunction(
    name="get_weather",
    description="Bir şehrin hava durumu bilgisini getirir",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Şehir adı (örn: İstanbul, Ankara)",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Sıcaklık birimi",
            },
        },
        "required": ["location"],
    },
)

# Calculator tool tanımı
calculator_tool = ToolFunction(
    name="calculator",
    description="Matematiksel hesaplama yapar",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Matematiksel ifade, örn: '15 + 25'",
            }
        },
        "required": ["expression"],
    },
)


def execute_tool(name: str, args: dict):
    """Tool execution callback."""
    print(f"🔧 Tool çağrısı: {name}({args})")

    if name == "get_weather":
        location = args.get("location", "")
        unit = args.get("unit", "celsius")
        # Mock hava durumu verisi
        mock_weather = {
            "istanbul": {"temp": 18, "condition": "Parçalı bulutlu"},
            "ankara": {"temp": 12, "condition": "Güneşli"},
            "izmir": {"temp": 22, "condition": "Açık"},
        }

        weather = mock_weather.get(location.lower(), {"temp": 15, "condition": "Bilinmiyor"})

        if unit == "fahrenheit":
            weather["temp"] = int(weather["temp"] * 9 / 5 + 32)
            unit_str = "°F"
        else:
            unit_str = "°C"

        return f"{location}: {weather['temp']}{unit_str}, {weather['condition']}"

    elif name == "calculator":
        expr = args.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Hesaplama hatası: {e}"

    return f"Bilinmeyen tool: {name}"


async def simple_weather_example():
    """Basit hava durumu örneği."""
    print("=" * 60)
    print("🌤️  VertexAI Gemini - Hava Durumu Tool Örneği")
    print("=" * 60)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT environment variable ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    try:
        provider = factory.create_vertexai(
            VertexAIConfig(
                project_id=project_id,
                location="us-central1",
                model="gemini-1.5-flash",
                temperature=0.1,
            )
        )
    except Exception as e:
        print(f"❌ Provider oluşturulamadı: {e}")
        return

    response = await run_with_tools_async(
        provider,
        prompt="İstanbul'un hava durumu nasıl?",
        tools=[weather_tool],
        execute_tool=execute_tool,
        max_iterations=2,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def calculator_example():
    """Hesap makinesi örneği."""
    print("=" * 60)
    print("🧮 VertexAI Gemini - Hesap Makinesi Tool Örneği")
    print("=" * 60)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT environment variable ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    try:
        provider = factory.create_vertexai(
            VertexAIConfig(
                project_id=project_id,
                location="us-central1",
                model="gemini-1.5-flash",
                temperature=0.1,
            )
        )
    except Exception as e:
        print(f"❌ Provider oluşturulamadı: {e}")
        return

    response = await run_with_tools_async(
        provider,
        prompt="(45 + 55) * 3 işleminin sonucu nedir?",
        tools=[calculator_tool],
        execute_tool=execute_tool,
        max_iterations=2,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def multi_tool_example():
    """Birden fazla tool örneği."""
    print("=" * 60)
    print("🔧 VertexAI Gemini - Multi-Tool Örneği")
    print("=" * 60)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT environment variable ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    try:
        provider = factory.create_vertexai(
            VertexAIConfig(
                project_id=project_id,
                location="us-central1",
                model="gemini-1.5-flash",
                temperature=0.2,
            )
        )
    except Exception as e:
        print(f"❌ Provider oluşturulamadı: {e}")
        return

    response = await run_with_tools_async(
        provider,
        prompt="İstanbul ve Ankara'nın sıcaklıklarının toplamı kaçtır? Celsius cinsinden.",
        tools=[weather_tool, calculator_tool],
        execute_tool=execute_tool,
        max_iterations=4,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def main():
    """Tüm örnekleri çalıştır."""
    print("\n🚀 VertexAI Gemini - Tools / Function Calling Examples\n")

    try:
        # Örnek 1: Hava durumu
        await simple_weather_example()

        # Örnek 2: Hesap makinesi
        await calculator_example()

        # Örnek 3: Birden fazla tool
        await multi_tool_example()

    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {e}")

    print("\n✨ Örnekler tamamlandı!")


if __name__ == "__main__":
    asyncio.run(main())
