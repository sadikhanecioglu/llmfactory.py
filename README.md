# LLM Provider Factory

A unified, extensible Python library for interacting with multiple Large Language Model (LLM) providers through a single, consistent interface. Support for OpenAI, Anthropic Claude, Google Gemini, VertexAI, and Ollama (local LLMs).

## 🌟 Features

- **Unified Interface**: Single API for multiple LLM providers
- **Cloud & Local LLMs**: Support for both cloud-based and local LLM providers
- **Async Support**: Full async/await support for better performance
- **Streaming**: Real-time streaming responses from all providers
- **Type Safety**: Complete type hints and Pydantic models
- **Error Handling**: Comprehensive error handling with specific exceptions
- **Configuration Management**: Flexible configuration system
- **Extensible**: Easy to add new providers
- **Testing**: Full test coverage with mocking support

## 🔌 Supported Providers

| Provider | Models | Features | Type |
|----------|--------|----------|------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4o | Generate, Stream, Conversation | Cloud |
| **Anthropic** | Claude-3 (Haiku, Sonnet, Opus) | Generate, Stream, Conversation | Cloud |
| **Google Gemini** | Gemini Pro, Gemini Flash | Generate, Stream, Conversation | Cloud |
| **VertexAI** | Mistral, Gemini | Generate, Conversation | Cloud |
| **Ollama** | Llama, CodeLlama, Mistral, etc. | Generate, Stream, Conversation | Local |

## 🚀 Quick Start

### Installation

```bash
pip install llm-provider-factory
```

### Basic Usage

```python
import asyncio
from llm_provider import LLMProviderFactory, OpenAIConfig, OllamaConfig

async def main():
    # Cloud LLM - OpenAI
    openai_config = OpenAIConfig(api_key="your-api-key", model="gpt-4")
    openai_provider = LLMProviderFactory().create_provider("openai", openai_config)
    response = await openai_provider.generate("Hello, world!")
    print(f"OpenAI: {response.content}")
    
    # Local LLM - Ollama
    ollama_config = OllamaConfig(
        base_url="http://localhost:11434",
        model="llama3.1:latest"
    )
    ollama_provider = LLMProviderFactory().create_provider("ollama", ollama_config)
    response = await ollama_provider.generate("Hello, world!")
    print(f"Ollama: {response.content}")

asyncio.run(main())
```

## 📖 Detailed Usage

### Configuration

#### Environment Variables
```bash
# Cloud LLM Providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Local LLM Providers
export OLLAMA_BASE_URL="http://localhost:11434"  # Default Ollama server
```

#### Programmatic Configuration
```python
from llm_provider import OpenAIConfig, AnthropicConfig, GeminiConfig, VertexAIConfig

# OpenAI Configuration
openai_config = OpenAIConfig(
    api_key="your-key",
    model="gpt-4",
    max_tokens=1000,
    temperature=0.7
)

# Anthropic Configuration
anthropic_config = AnthropicConfig(
    api_key="your-key",
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    temperature=0.7
)

# Gemini Configuration
gemini_config = GeminiConfig(
    api_key="your-key",
    model="gemini-pro",
    max_tokens=1000,
    temperature=0.7
)

# VertexAI Configuration  
vertexai_config = VertexAIConfig(
    project_id="your-gcp-project",
    location="us-central1",
    model="gemini-1.5-pro",
    credentials_path="path/to/service-account.json"  # Optional if using GOOGLE_APPLICATION_CREDENTIALS
)

# Ollama Configuration (Local LLM)
ollama_config = OllamaConfig(
    base_url="http://localhost:11434",  # Default Ollama server
    model="llama3.1:latest",  # Any Ollama model
    max_tokens=1000,
    temperature=0.7
)
```

### Multiple Provider Usage

```python
from llm_provider import LLMProviderFactory

async def compare_providers():
    factory = LLMProviderFactory()
    prompt = "Explain quantum computing in simple terms"
    
    # Generate with different providers
    openai_response = await factory.generate(prompt, provider="openai")
    anthropic_response = await factory.generate(prompt, provider="anthropic")
    gemini_response = await factory.generate(prompt, provider="gemini")
    vertexai_response = await factory.generate(prompt, provider="vertexai")
    
    print(f"OpenAI: {openai_response.content}")
    print(f"Anthropic: {anthropic_response.content}")
    print(f"Gemini: {gemini_response.content}")
    print(f"VertexAI: {vertexai_response.content}")
```

### Conversation History

```python
from llm_provider import Message, MessageRole

async def conversation_example():
    factory = LLMProviderFactory.create_openai()
    
    history = [
        Message(role=MessageRole.USER, content="Hello, I'm learning Python"),
        Message(role=MessageRole.ASSISTANT, content="Hello! I'd be happy to help you learn Python."),
        Message(role=MessageRole.USER, content="Can you explain variables?")
    ]
    
    response = await factory.generate(
        "Now explain functions",
        history=history
    )
    print(response.content)
```

### Streaming Responses

```python
async def streaming_example():
    factory = LLMProviderFactory.create_openai()
    
    async for chunk in factory.stream_generate("Write a short story about AI"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        
        if chunk.is_final:
            print(f"\nFinish reason: {chunk.finish_reason}")
            break
```

### Error Handling

```python
from llm_provider import (
    AuthenticationError,
    RateLimitError,
    ModelNotAvailableError,
    GenerationError
)

async def robust_generation():
    factory = LLMProviderFactory.create_openai()
    
    try:
        response = await factory.generate("Hello world")
        return response.content
    except AuthenticationError:
        print("Check your API key")
    except RateLimitError:
        print("Rate limit exceeded, try again later")
    except ModelNotAvailableError:
        print("Model not available")
    except GenerationError as e:
        print(f"Generation failed: {e}")
```

## 🔧 Advanced Usage

### Custom Provider

```python
from llm_provider import BaseLLMProvider, ProviderConfig

class CustomProvider(BaseLLMProvider):
    async def initialize(self):
        # Initialize your custom provider
        pass
    
    async def generate(self, request):
        # Implement generation logic
        pass
    
    async def stream_generate(self, request):
        # Implement streaming logic
        pass
    
    def get_supported_models(self):
        return ["custom-model-1", "custom-model-2"]
    
    def validate_config(self):
        return True
    
    def get_provider_info(self):
        # Return provider information
        pass

# Register custom provider
factory = LLMProviderFactory()
factory.register_provider("custom", CustomProvider)
```

### Provider Information

```python
async def provider_info_example():
    factory = LLMProviderFactory()
    
    # Get all provider information
    all_providers = factory.get_provider_info()
    for info in all_providers:
        print(f"{info.display_name}: {info.supported_models}")
    
    # Get specific provider info
    openai_info = factory.get_provider_info("openai")
    print(f"OpenAI models: {openai_info.supported_models}")
```

## 📁 Project Structure

```
llm-provider-factory/
├── src/
│   └── llm_provider/
│       ├── __init__.py
│       ├── factory.py
│       ├── base_provider.py
│       ├── settings.py
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   └── gemini_provider.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── exceptions.py
│           └── logger.py
├── tests/
├── pyproject.toml
└── README.md
```

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/llm_provider --cov-report=html

# Run specific test file
pytest tests/test_factory.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-provider`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Submit a pull request

### Adding a New Provider

1. Create a new provider class in `src/llm_provider/providers/`
2. Inherit from `BaseLLMProvider`
3. Implement all abstract methods
4. Add configuration class in `utils/config.py`
5. Register the provider in `factory.py`
6. Add tests in `tests/`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for their excellent API and documentation
- Anthropic for Claude's capabilities
- Google for Gemini's multimodal features
- The Python community for inspiration and tools

# LLM Provider Factory

A unified factory for multiple LLM providers (OpenAI, Anthropic, Google Gemini, Google Vertex AI).

## VertexAI + Mistral Support

Bu paket artık Google Cloud Vertex AI üzerinden Mistral modellerini desteklemektedir!

### Desteklenen Modeller

**VertexAI Provider:**
- Gemini modelleri: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.0-pro`
- Text/Chat modelleri: `text-bison`, `chat-bison` 
- **Mistral modelleri**: `mistral-large-2411`, `mistral-7b-instruct`

### Hızlı Başlangıç - VertexAI

1. **Paketleri yükleyin:**
```bash
python setup_vertexai.py
```

2. **Google Cloud Setup:**
   - Service account oluşturun
   - JSON credentials dosyası indirin
   - Environment variable ayarlayın:
```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/credentials.json'
```

3. **Kullanım:**
```python
from llm_provider import LLMProviderFactory
from llm_provider.utils.config import VertexAIConfig

# Configuration
config = VertexAIConfig(
    project_id="your-project-id",
    location="us-central1", 
    model="mistral-large-2411",  # Mistral model!
    credentials_path="/path/to/credentials.json",
    temperature=0.1,
    max_tokens=1000
)

# Create provider
factory = LLMProviderFactory()
provider = factory.create_provider("vertexai", config)

# Generate response
response = await provider.generate(request)
```

4. **Test:**
```bash
python test_vertexai_mistral.py
``` Built with clean architecture principles and SOLID design patterns.

## 🚀 Quick Start

```bash
pip install llm-provider
```

```python
from llm_provider import LLMProviderFactory, OpenAI

provider = LLMProviderFactory(OpenAI(api_key="your-key"))
response = provider.generate(prompt="Hello", history=[])
print(response.content)
```

## ✨ Features

- 🏭 **Factory Pattern**: Clean, consistent interface
- 🔌 **Extensible**: Easy to add new providers  
- 🛡️ **Type Safe**: Full typing support
- 🚀 **Production Ready**: Comprehensive error handling
- 📦 **Zero Dependencies**: Only requires `requests`

## 🔗 Links

- **PyPI**: https://pypi.org/project/llm-provider/
- **Test PyPI**: https://test.pypi.org/project/llm-provider/

## 📦 Supported Providers

- **OpenAI** (GPT-3.5, GPT-4)
- **Anthropic** (Claude models)
- **Google Gemini** (Gemini Pro, Flash)

## 📚 Documentation

See the package source code and examples in the repository.