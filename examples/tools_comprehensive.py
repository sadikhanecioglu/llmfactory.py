"""
Comprehensive Tools / Function Calling Example

Bu örnek şu özellikleri gösterir:
- Tool tanımlama (ToolFunction)
- OpenAI function calling desteği
- run_with_tools ile otomatik tool execution loop
- Matematiksel hesaplama ve bilgi arama tool'ları
"""

import asyncio
import os
from llm_provider import (
    LLMProviderFactory,
    OpenAIConfig,
    ToolFunction,
    Message,
    MessageRole,
)
from llm_provider.utils import run_with_tools_async


# Tool 1: Hesap makinesi
calculator_tool = ToolFunction(
    name="calculator",
    description="Matematiksel işlemleri hesaplar (toplama, çıkarma, çarpma, bölme)",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Matematiksel ifade, örn: '(12 + 30) * 2'",
            }
        },
        "required": ["expression"],
    },
)

# Tool 2: Bilgi arama (mock)
search_tool = ToolFunction(
    name="web_search",
    description="İnternetten bilgi arar",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Arama sorgusu",
            }
        },
        "required": ["query"],
    },
)


def execute_tool(name: str, args: dict):
    """Tool execution callback."""
    print(f"🔧 Executing tool: {name} with args: {args}")

    if name == "calculator":
        expr = args.get("expression", "")
        try:
            # UYARI: Gerçek uygulamada daha güvenli bir parser kullanın
            result = eval(expr, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Hesaplama hatası: {e}"

    elif name == "web_search":
        query = args.get("query", "")
        # Mock arama sonucu
        mock_results = {
            "python": "Python, yüksek seviyeli bir programlama dilidir. 1991'de Guido van Rossum tarafından geliştirildi.",
            "istanbul": "İstanbul, Türkiye'nin en kalabalık şehridir. 15+ milyon nüfusu vardır.",
        }

        for keyword, info in mock_results.items():
            if keyword.lower() in query.lower():
                return info
        return f"'{query}' hakkında sonuç bulunamadı."

    return f"Bilinmeyen tool: {name}"


async def simple_example():
    """Basit hesap makinesi örneği."""
    print("=" * 60)
    print("📱 Basit Hesap Makinesi Örneği")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    provider = factory.create_openai(
        OpenAIConfig(api_key=api_key, model="gpt-4o-mini")
    )

    response = await run_with_tools_async(
        provider,
        prompt="(15 + 25) * 4 işleminin sonucu nedir? Hesap makinesini kullan.",
        tools=[calculator_tool],
        execute_tool=execute_tool,
        max_iterations=3,
        temperature=0.1,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def multi_tool_example():
    """Birden fazla tool ile örnek."""
    print("=" * 60)
    print("📚 Multi-Tool Örneği")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    provider = factory.create_openai(
        OpenAIConfig(api_key=api_key, model="gpt-4o-mini")
    )

    response = await run_with_tools_async(
        provider,
        prompt="Python hakkında bilgi bul ve Python'un çıkış yılı ile 2024 arasındaki farkı hesapla.",
        tools=[calculator_tool, search_tool],
        execute_tool=execute_tool,
        max_iterations=5,
        temperature=0.2,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def conversation_with_tools():
    """Konuşma geçmişi ile tool kullanımı."""
    print("=" * 60)
    print("💬 Konuşma + Tool Örneği")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY ayarlanmamış!")
        return

    factory = LLMProviderFactory()
    provider = factory.create_openai(
        OpenAIConfig(api_key=api_key, model="gpt-4o-mini")
    )

    # İlk mesajlar
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="Sen yardımcı bir asistansın. Kullanıcıya matematik ve bilgi arama konularında yardımcı ol.",
        ),
        Message(
            role=MessageRole.USER, content="Merhaba! Matematiksel hesaplamalar yapabilir misin?"
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Evet, hesap makinesini kullanarak matematiksel işlemler yapabilirim.",
        ),
        Message(role=MessageRole.USER, content="100 ile 50'nin toplamı kaçtır?"),
    ]

    response = await run_with_tools_async(
        provider,
        messages=messages,
        tools=[calculator_tool],
        execute_tool=execute_tool,
        max_iterations=2,
        temperature=0.1,
    )

    print(f"\n🤖 Cevap: {response.content}\n")


async def main():
    """Tüm örnekleri çalıştır."""
    print("\n🧰 LLM Provider Factory - Tools / Function Calling Examples\n")

    try:
        # Örnek 1: Basit hesap
        await simple_example()

        # Örnek 2: Birden fazla tool
        await multi_tool_example()

        # Örnek 3: Konuşma geçmişi
        await conversation_with_tools()

    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {e}")

    print("\n✨ Örnekler tamamlandı!")


if __name__ == "__main__":
    asyncio.run(main())
