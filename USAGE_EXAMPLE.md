# llm-provider-factory v0.7.0 - Kullanım Örnekleri

## 📦 Kurulum

```bash
# PyPI'dan kurulum (yayınlandıktan sonra)
pip install llm-provider-factory

# Veya local build'den kurulum
pip install dist/llm_provider_factory-0.7.0-py3-none-any.whl

# Kaynak koddan kurulum
pip install -e .
```

## 🆕 v0.7.0 Yenilikleri

### ✨ Tools (Function Calling) Desteği

Bu versiyonda OpenAI ve VertexAI Gemini modelleri için **function calling** desteği eklendi!

---

## 🔧 Hızlı Başlangıç

### 1. OpenAI ile Tools Kullanımı

```python
import asyncio
from llm_provider import (
    LLMProviderFactory, 
    OpenAIConfig, 
    ToolFunction
)
from llm_provider.utils import run_with_tools_async

# Hesap makinesi tool'u tanımla
calculator = ToolFunction(
    name="calculator",
    description="Matematiksel ifadeleri hesaplar",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Hesaplanacak matematiksel ifade (örn: '12+30*2')"
            }
        },
        "required": ["expression"]
    }
)

# Tool'u çalıştıracak fonksiyon
def execute_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            # Güvenli eval kullanımı
            result = eval(args["expression"], {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Hata: {e}"
    return "Bilinmeyen tool"

async def main():
    # Provider oluştur
    config = OpenAIConfig(
        api_key="your-openai-api-key",
        model="gpt-4o-mini"
    )
    provider = LLMProviderFactory().create_openai(config)
    
    # Tools ile çalıştır
    response = await run_with_tools_async(
        provider=provider,
        prompt="(15 + 25) * 3 işlemini hesapla",
        tools=[calculator],
        execute_tool=execute_tool,
        max_iterations=3
    )
    
    print(f"Sonuç: {response.content}")
    print(f"Token kullanımı: {response.usage}")

# Çalıştır
asyncio.run(main())
```

**Çıktı:**
```
Sonuç: İşlemin sonucu 120'dir.
Token kullanımı: {'prompt_tokens': 145, 'completion_tokens': 28, 'total_tokens': 173}
```

---

### 2. VertexAI Gemini ile Tools

```python
import asyncio
from llm_provider import (
    LLMProviderFactory,
    VertexAIConfig,
    ToolFunction
)
from llm_provider.utils import run_with_tools_async

# Hava durumu tool'u
weather_tool = ToolFunction(
    name="get_weather",
    description="Belirtilen şehir için hava durumu bilgisi döndürür",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Şehir ismi (örn: Istanbul, Ankara, Izmir)"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Sıcaklık birimi"
            }
        },
        "required": ["city"]
    }
)

# Döviz kurları tool'u
currency_tool = ToolFunction(
    name="get_currency_rate",
    description="İki para birimi arasındaki döviz kurunu döndürür",
    parameters={
        "type": "object",
        "properties": {
            "from_currency": {
                "type": "string",
                "description": "Kaynak para birimi (örn: USD, EUR, TRY)"
            },
            "to_currency": {
                "type": "string",
                "description": "Hedef para birimi"
            }
        },
        "required": ["from_currency", "to_currency"]
    }
)

def execute_tool(name: str, args: dict) -> str:
    """Tool'ları çalıştır"""
    if name == "get_weather":
        city = args.get("city", "")
        unit = args.get("unit", "celsius")
        # Gerçek uygulamada API çağrısı yapılır
        temp = "22" if unit == "celsius" else "72"
        symbol = "°C" if unit == "celsius" else "°F"
        return f"{city} için hava durumu: {temp}{symbol}, Güneşli"
    
    elif name == "get_currency_rate":
        from_curr = args.get("from_currency")
        to_curr = args.get("to_currency")
        # Gerçek uygulamada API çağrısı yapılır
        rates = {"USD-TRY": 33.50, "EUR-TRY": 36.20, "USD-EUR": 0.92}
        rate_key = f"{from_curr}-{to_curr}"
        rate = rates.get(rate_key, 1.0)
        return f"1 {from_curr} = {rate} {to_curr}"
    
    return "Bilinmeyen tool"

async def main():
    # VertexAI provider oluştur
    config = VertexAIConfig(
        project_id="your-project-id",
        location="us-central1",
        model="gemini-2.0-flash-exp"
    )
    provider = LLMProviderFactory().create_vertexai(config)
    
    # Çoklu tool kullanımı
    response = await run_with_tools_async(
        provider=provider,
        prompt="İstanbul'da hava nasıl ve 1 USD kaç TL?",
        tools=[weather_tool, currency_tool],
        execute_tool=execute_tool,
        max_iterations=5
    )
    
    print(f"Sonuç:\n{response.content}")

asyncio.run(main())
```

**Çıktı:**
```
Sonuç:
İstanbul'da hava durumu güneşli ve 22°C. Döviz kuru açısından, 1 USD = 33.50 TRY.
```

---

### 3. Mesaj Geçmişi ile Tools

```python
import asyncio
from llm_provider import (
    LLMProviderFactory,
    OpenAIConfig,
    Message,
    MessageRole,
    ToolFunction
)
from llm_provider.utils import run_with_tools_async

# Web arama tool'u (mock)
search_tool = ToolFunction(
    name="web_search",
    description="İnternette arama yapar ve sonuçları döndürür",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Arama sorgusu"
            }
        },
        "required": ["query"]
    }
)

def execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        query = args.get("query", "")
        # Mock sonuç
        return f"'{query}' için arama sonuçları: Python 3.12, 2023'te yayınlandı..."
    return "Bilinmeyen tool"

async def main():
    config = OpenAIConfig(
        api_key="your-openai-api-key",
        model="gpt-4o-mini"
    )
    provider = LLMProviderFactory().create_openai(config)
    
    # Konuşma geçmişi ile başla
    messages = [
        Message(role=MessageRole.SYSTEM, content="Sen yardımcı bir asistansın."),
        Message(role=MessageRole.USER, content="Python'un son sürümü nedir?")
    ]
    
    response = await run_with_tools_async(
        provider=provider,
        messages=messages,
        tools=[search_tool],
        execute_tool=execute_tool,
        max_iterations=3
    )
    
    print(f"Asistan: {response.content}")

asyncio.run(main())
```

---

### 4. Manuel Tool Kullanımı (Low-Level)

Otomatik loop kullanmadan manuel tool yönetimi:

```python
import asyncio
from llm_provider import (
    LLMProviderFactory,
    OpenAIConfig,
    GenerationRequest,
    Message,
    MessageRole,
    ToolFunction
)

async def main():
    # Provider oluştur
    config = OpenAIConfig(api_key="your-key", model="gpt-4o-mini")
    provider = LLMProviderFactory().create_openai(config)
    
    # Tool tanımla
    calculator = ToolFunction(
        name="calculator",
        description="Matematiksel hesaplamalar yapar",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    )
    
    # İlk istek
    request = GenerationRequest(
        messages=[
            Message(role=MessageRole.USER, content="15 + 27 kaç eder?")
        ],
        tools=[calculator],
        tool_choice="auto"
    )
    
    response = await provider.generate(request)
    
    # Tool çağrısı var mı?
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool çağrısı: {tool_call.name}")
            print(f"Parametreler: {tool_call.arguments}")
            
            # Tool'u çalıştır
            result = "42"  # Hesaplama sonucu
            
            # Sonucu geri gönder
            request.messages.append(
                Message(role=MessageRole.ASSISTANT, content=response.content)
            )
            request.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result,
                    tool_call_id=tool_call.id,
                    name=tool_call.name
                )
            )
            
            # Tekrar sor
            final_response = await provider.generate(request)
            print(f"Sonuç: {final_response.content}")

asyncio.run(main())
```

---

## 🎯 Diğer Provider'lar

### OpenAI (Normal Kullanım)

```python
import asyncio
from llm_provider import LLMProviderFactory, OpenAIConfig

async def main():
    config = OpenAIConfig(
        api_key="your-key",
        model="gpt-4o-mini",
        temperature=0.7
    )
    provider = LLMProviderFactory().create_openai(config)
    
    response = await provider.generate_response(
        "Python'da async/await nasıl çalışır?",
        max_tokens=500
    )
    
    print(response.content)

asyncio.run(main())
```

### Anthropic Claude

```python
import asyncio
from llm_provider import LLMProviderFactory, AnthropicConfig

async def main():
    config = AnthropicConfig(
        api_key="your-key",
        model="claude-3-5-sonnet-20241022"
    )
    provider = LLMProviderFactory().create_anthropic(config)
    
    response = await provider.generate_response(
        "Kuantum hesaplama nedir?",
        temperature=0.5
    )
    
    print(response.content)

asyncio.run(main())
```

### Gemini (Google AI)

```python
import asyncio
from llm_provider import LLMProviderFactory, GeminiConfig

async def main():
    config = GeminiConfig(
        api_key="your-key",
        model="gemini-2.0-flash-exp"
    )
    provider = LLMProviderFactory().create_gemini(config)
    
    response = await provider.generate_response(
        "Yapay zeka etiği hakkında düşüncelerin neler?"
    )
    
    print(response.content)

asyncio.run(main())
```

### Ollama (Local)

```python
import asyncio
from llm_provider import LLMProviderFactory, OllamaConfig

async def main():
    config = OllamaConfig(
        base_url="http://localhost:11434",
        model="llama3.2"
    )
    provider = LLMProviderFactory().create_ollama(config)
    
    response = await provider.generate_response(
        "Merhaba, nasılsın?"
    )
    
    print(response.content)

asyncio.run(main())
```

---

## 🎨 Image Generation

```python
import asyncio
from llm_provider import ImageFactory, OpenAIImageConfig

async def main():
    config = OpenAIImageConfig(
        api_key="your-key",
        model="dall-e-3"
    )
    provider = ImageFactory().create_openai(config)
    
    result = await provider.generate_image(
        prompt="Gün batımında dağların üzerinde uçan renkli balonlar",
        size="1024x1024",
        quality="hd",
        style="vivid"
    )
    
    print(f"Görsel URL: {result.url}")
    # Veya base64: result.b64_json

asyncio.run(main())
```

---

## 🎤 Speech-to-Text

```python
import asyncio
from llm_provider import SpeechFactory, OpenAISpeechConfig, SpeechRequest

async def main():
    config = OpenAISpeechConfig(api_key="your-key")
    provider = SpeechFactory().create_openai(config)
    
    request = SpeechRequest(
        audio_data="/path/to/audio.mp3",
        language="tr",  # Türkçe
        response_format="json"
    )
    
    response = await provider.transcribe(request)
    print(f"Transkript: {response.text}")

asyncio.run(main())
```

---

## 📊 Streaming Desteği

```python
import asyncio
from llm_provider import LLMProviderFactory, OpenAIConfig

async def main():
    config = OpenAIConfig(api_key="your-key", model="gpt-4o-mini")
    provider = LLMProviderFactory().create_openai(config)
    
    print("Cevap: ", end="", flush=True)
    
    async for chunk in provider.generate_stream("Python hakkında 3 cümle yaz"):
        print(chunk.content, end="", flush=True)
    
    print()  # Yeni satır

asyncio.run(main())
```

---

## 🔒 Hata Yönetimi

```python
import asyncio
from llm_provider import LLMProviderFactory, OpenAIConfig
from llm_provider.utils.exceptions import (
    ProviderAPIError,
    ConfigurationError,
    RateLimitError
)

async def main():
    try:
        config = OpenAIConfig(api_key="invalid-key")
        provider = LLMProviderFactory().create_openai(config)
        
        response = await provider.generate_response("Test")
        
    except RateLimitError as e:
        print(f"Rate limit aşıldı: {e}")
    except ProviderAPIError as e:
        print(f"API hatası: {e}")
    except ConfigurationError as e:
        print(f"Yapılandırma hatası: {e}")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")

asyncio.run(main())
```

---

## 🚀 İleri Düzey Kullanım

### Environment Variables ile Yapılandırma

```python
# .env dosyası
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
# VERTEX_PROJECT_ID=my-project
# VERTEX_LOCATION=us-central1

from llm_provider import LLMProviderFactory
import os

# Environment'tan otomatik yükle
factory = LLMProviderFactory()

# OpenAI
openai_provider = factory.create_openai()

# Anthropic
anthropic_provider = factory.create_anthropic()

# Gemini
gemini_provider = factory.create_gemini()
```

### Timeout ve Retry Yapılandırması

```python
from llm_provider import OpenAIConfig

config = OpenAIConfig(
    api_key="your-key",
    model="gpt-4o-mini",
    timeout=30.0,  # 30 saniye timeout
    max_retries=3   # 3 deneme
)
```

---

## 📝 Notlar

1. **API Keys**: Tüm örneklerde `"your-key"` yerine gerçek API anahtarlarınızı kullanın
2. **Async/Await**: Tüm provider metodları async olduğu için `asyncio.run()` gereklidir
3. **Tools**: Function calling özelliği şu an OpenAI ve VertexAI Gemini'de desteklenmektedir
4. **Rate Limits**: Provider'ların rate limit'lerine dikkat edin
5. **Costs**: Tool çağrıları ekstra token kullanımına neden olabilir

---

## 🔗 Faydalı Linkler

- **GitHub**: [sadikhanecioglu/llmfactory.py](https://github.com/sadikhanecioglu/llmfactory.py)
- **Daha fazla örnek**: `examples/` klasörüne bakın
- **Testler**: `tests/` klasöründe birim testler mevcut

---

## 📦 Paket Bilgileri

- **Version**: 0.7.0
- **Python**: >=3.8
- **License**: MIT
- **Build**: `dist/llm_provider_factory-0.7.0-py3-none-any.whl`

---

## 🎉 Yeni Başlayanlar İçin Hızlı Test

```python
import asyncio
from llm_provider import LLMProviderFactory, OpenAIConfig

async def quick_test():
    # 1. Basit kullanım
    config = OpenAIConfig(api_key="your-key", model="gpt-4o-mini")
    provider = LLMProviderFactory().create_openai(config)
    
    response = await provider.generate_response("Merhaba!")
    print(response.content)

asyncio.run(quick_test())
```

Bu dosyayı kaydedin ve `your-key` yerine gerçek API anahtarınızı yazarak çalıştırın! 🚀
