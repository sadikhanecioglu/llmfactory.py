"""Google Cloud Vertex AI/Gen AI provider implementation."""

from typing import Optional, AsyncIterator, List, Dict, Any
import os
import httpx

from ..base_provider import BaseLLMProvider
from ..settings import (
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    ProviderInfo,
    ToolFunction,
    ToolCall,
)
from ..utils.config import VertexAIConfig
from ..utils.exceptions import (
    InvalidConfigurationError,
    AuthenticationError,
    APIError,
    GenerationError,
)
from ..utils.logger import logger

# Import Google Gen AI SDK with proper error handling
try:
    # YENİ: Google Gen AI SDK - sadece Client'ı kullan
    from google.genai import Client
    from google.genai import types as genai_types
    import google.auth
    from google.auth.transport.requests import Request

    GENAI_AVAILABLE = True
except ImportError as e:
    GENAI_AVAILABLE = False
    genai_types = None
    Client = None
    logger.warning(f"⚠️ google-genai import hatası: {e}")


class VertexAIProvider(BaseLLMProvider):
    """Google Cloud Vertex AI/Gen AI LLM sağlayıcısı (Gemini ve Mistral modelleri)"""

    SUPPORTED_MODELS = [
        "gemini-2.0-flash-001",  # 🔥 Gemini 2.0 Flash (yeni)
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "text-bison",
        "text-bison-32k",
        "chat-bison",
        "chat-bison-32k",
        "mistral-large-2411",
        "mistral-7b-instruct",
    ]

    def __init__(self, config: Optional[VertexAIConfig] = None) -> None:
        """Initialize Vertex AI/Gen AI provider."""
        if config is None:
            config = VertexAIConfig.from_env()

        super().__init__(config)
        self.config: VertexAIConfig = config
        self.provider_name = "vertexai"
        self.project_id = config.project_id
        self.location = config.location
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_tokens
        self.client: Optional[Client] = None  # Gen AI Client objesi

        # Set credentials if provided
        if config.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path

        logger.info(
            f"🔧 Gen AI Provider oluşturuldu: model={self.model_name}, project={self.project_id}"
        )

    async def initialize(self) -> None:
        """Initialize Vertex AI client."""
        if not GENAI_AVAILABLE:
            raise InvalidConfigurationError(
                "Google GenAI paketi yüklenmemiş.", "vertexai"
            )

        if not self.project_id:
            raise InvalidConfigurationError(
                "Google Cloud Project ID gereklidir.", "vertexai"
            )

        try:
            # Google Gen AI Client kullan (yeni API)
            from google import genai

            self.client = genai.Client(
                vertexai=True, project=self.project_id, location=self.location
            )

            logger.info(
                f"✅ Gen AI Client başlatıldı: {self.project_id}, location: {self.location}, model: {self.model_name}"
            )

        except Exception as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                raise AuthenticationError(
                    f"Vertex AI authentication failed: {str(e)}", "vertexai"
                )
            else:
                raise APIError(
                    f"Failed to initialize Vertex AI client: {str(e)}", "vertexai"
                )

    def validate_config(self) -> bool:
        """Validate Gen AI configuration."""
        if not self.config.project_id:
            raise InvalidConfigurationError(
                "Google Cloud Project ID gereklidir", "genai"
            )

        if self.config.model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{self.config.model}' tam desteklenmeyebilir. "
                f"Desteklenenler: {', '.join(self.SUPPORTED_MODELS)}"
            )
        return True

    def is_available(self) -> bool:
        """Sağlayıcı kullanılabilir mi?"""
        return (
            GENAI_AVAILABLE
            and self.model_name is not None
            and self.project_id is not None
        )

    # Mistral için kullanılan yardımcı fonksiyonlar (değişmedi)
    # _get_credentials_token ve _build_mistral_endpoint_url olduğu gibi kalabilir.
    # ...

    def _convert_messages_to_genai(
        self, messages: List[Dict], system_prompt: Optional[str] = None
    ) -> List[Any]:
        """Convert a list of generic message dicts to Gen AI SDK Content objects."""
        # DİKKAT: Bu fonksiyon artık `generate` tarafından doğrudan kullanılmıyor,
        # ancak `stream_generate` içindeki eski mantık için tutulabilir.
        # Yeni `generate` metodu içeriği doğrudan kendisi oluşturur.
        genai_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role in ["user", "tool"]:
                genai_role = "user"
            elif role in ["assistant"]:
                genai_role = "model"
            else:
                genai_role = "user"
            genai_messages.append(
                genai_types.Content(
                    role=genai_role, parts=[genai_types.Part(text=content)]
                )
            )
        return genai_messages

    def _convert_tools_to_genai(
        self, tools: Optional[List[ToolFunction]]
    ) -> Optional[List[Any]]:
        """Convert ToolFunction list to Gemini function declarations."""
        if not tools or not GENAI_AVAILABLE:
            return None

        function_declarations = []
        for tool in tools:
            func_decl = genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.parameters or {"type": "object", "properties": {}},
            )
            function_declarations.append(func_decl)

        if not function_declarations:
            return None
        return [genai_types.Tool(function_declarations=function_declarations)]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using VertexAI/GenAI.

        Args:
            request: The generation request

        Returns:
            Generated response

        Raises:
            GenerationError: If generation fails
        """
        try:
            await self.ensure_initialized()

            history_messages = []
            if request.history:
                for msg in request.history:
                    history_messages.append(
                        {
                            "role": (
                                str(msg.role)
                                if hasattr(msg.role, "value")
                                else str(msg.role)
                            ),
                            "content": msg.content,
                        }
                    )

            # Mistral modelleri için özel rawPredict API kullan
            if "mistral" in self.model_name.lower():
                # Mistral için mesajları hazırla
                mistral_messages = []
                system_prompt = getattr(request, "system_prompt", None)
                if system_prompt:
                    mistral_messages.append(
                        {"role": "system", "content": system_prompt}
                    )

                if history_messages:
                    for msg in history_messages:
                        mistral_messages.append(
                            {
                                "role": msg.get("role", "user"),
                                "content": msg.get("content", ""),
                            }
                        )
                mistral_messages.append({"role": "user", "content": request.prompt})

                # _generate_mistral_response
                response_text = await self._generate_mistral_response(mistral_messages)

                # Mistral (rawPredict) API'si token sayısını döndürmez
                prompt_tokens = sum(len(m["content"].split()) for m in mistral_messages)
                completion_tokens = len(response_text.split())
                total_tokens = prompt_tokens + completion_tokens
                tool_calls = None  # Mistral için tool desteği yok

            else:
                # Gemini modelleri için standart Gen AI Client API

                # 1. Mesajları (içerik) hazırla
                contents = []
                system_instruction = getattr(request, "system_prompt", None)

                if history_messages:
                    for msg in history_messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if not content or role == "system":
                            continue
                        genai_role = (
                            "model" if role in ["assistant", "model"] else "user"
                        )
                        contents.append(
                            genai_types.Content(
                                role=genai_role, parts=[genai_types.Part(text=content)]
                            )
                        )

                if request.prompt:
                    contents.append(
                        genai_types.Content(
                            role="user", parts=[genai_types.Part(text=request.prompt)]
                        )
                    )

                # 2. Araçları (tools) hazırla
                genai_tools = (
                    self._convert_tools_to_genai(request.tools)
                    if request.tools
                    else None
                )

                # 3. Config - dictionary olarak (google-genai yeni API)
                config_dict = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }

                logger.info(
                    f"🔍 DEBUG - request.tools type: {type(request.tools)}, value: {request.tools}"
                )
                logger.info(
                    f"🔍 DEBUG - genai_tools type: {type(genai_tools)}, value: {genai_tools}"
                )
                logger.info(
                    f"🔍 DEBUG - config_dict type: {type(config_dict)}, value: {config_dict}"
                )

                logger.info(
                    f"Gen AI'ya gönderiliyor: {len(contents)} content, "
                    f"tools: {len(request.tools) if request.tools else 0}, "
                    f"system: {'Yes' if system_instruction else 'No'}"
                )

                # 4. API'yi çağır
                generate_kwargs = {
                    "model": self.model_name,
                    "contents": contents,
                    "config": config_dict,
                }

                # 🔥 IMPORTANT: Google Gen AI uses 'config' dict that can include 'tools'
                # But actually tools should be passed separately to generate_content()
                # Let's check the actual API signature...
                if genai_tools:
                    logger.info(f"🔍 DEBUG - Adding tools to CONFIG (not kwargs): {genai_tools}")
                    # Try adding tools to config dict instead
                    config_dict["tools"] = genai_tools
                    generate_kwargs["config"] = config_dict

                if system_instruction:
                    logger.info(f"🔍 DEBUG - Adding system_instruction: {system_instruction[:100]}...")
                    generate_kwargs["system_instruction"] = system_instruction

                logger.info(f"🔍 DEBUG - Final generate_kwargs keys: {generate_kwargs.keys()}")
                logger.info(f"🔍 DEBUG - generate_kwargs['config']: {generate_kwargs.get('config')}")
                
                try:
                    response = self.client.models.generate_content(**generate_kwargs)
                except Exception as api_error:
                    logger.error(f"❌ API call failed with error: {type(api_error).__name__}: {api_error}")
                    logger.error(f"❌ generate_kwargs that caused error: {generate_kwargs}")
                    raise

                # 5. YANITI DOĞRU ŞEKİLDE İŞLE
                response_text = ""
                tool_calls_list: List[ToolCall] = []

                if not response.candidates:
                    raise GenerationError(
                        "No candidates returned from Gemini", "vertexai"
                    )

                first_candidate = response.candidates[0]

                # Önce function_calls varsa işle (örnekteki gibi)
                if (
                    hasattr(first_candidate, "function_calls")
                    and first_candidate.function_calls
                ):
                    logger.info(
                        f"🔧 Function calls detected: {len(first_candidate.function_calls)}"
                    )

                    for function_call in first_candidate.function_calls:
                        logger.info(f"Function call: {function_call.name}")

                        # Arguments'i dict'e çevir
                        tool_call_args = {}
                        if hasattr(function_call, "args") and function_call.args:
                            tool_call_args = dict(function_call.args)

                        tool_calls_list.append(
                            ToolCall(
                                id=f"tool_call_{function_call.name}_{len(tool_calls_list)}",
                                name=function_call.name,
                                arguments=tool_call_args,
                            )
                        )
                else:
                    # Text response varsa al
                    for part in first_candidate.content.parts:
                        if part.text:
                            response_text += part.text

                tool_calls = tool_calls_list if tool_calls_list else None
                response_text = response_text.strip()

                # Kullanım (usage) bilgilerini al - usage_metadata bir obje
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_metadata = response.usage_metadata
                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                    completion_tokens = getattr(
                        usage_metadata, "candidates_token_count", 0
                    )
                    total_tokens = getattr(usage_metadata, "total_token_count", 0)

            # (Mistral ve Gemini blokları burada birleşir)
            return GenerationResponse(
                content=(
                    response_text if response_text else None
                ),  # Tool call varsa content None olabilir
                provider=self.provider_name,
                model=self.model_name,
                tool_calls=tool_calls,  # Düzeltildi
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )

        except Exception as e:
            logger.error(f"VertexAI generation failed: {e}")
            raise GenerationError(f"VertexAI generation failed: {str(e)}", "vertexai")

    async def stream_generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream generate response using Gen AI."""
        try:
            # Gemini/Gen AI için GERÇEK streaming metodu
            if "mistral" not in self.model_name.lower():
                await self.ensure_initialized()

                # 1. Mesajları (içerik) hazırla
                contents = []
                system_instruction = getattr(request, "system_prompt", None)

                if request.history:
                    for msg in request.history:
                        role = (
                            str(msg.role)
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        )
                        content = msg.content
                        if not content or role == "system":
                            continue
                        genai_role = (
                            "model" if role in ["assistant", "model"] else "user"
                        )
                        contents.append(
                            genai_types.Content(
                                role=genai_role, parts=[genai_types.Part(text=content)]
                            )
                        )

                if request.prompt:
                    contents.append(
                        genai_types.Content(
                            role="user", parts=[genai_types.Part(text=request.prompt)]
                        )
                    )

                if not contents:
                    raise GenerationError(
                        "Prompt or history must be provided for streaming", "genai"
                    )

                # 2. Araçları (tools) hazırla
                genai_tools = (
                    self._convert_tools_to_genai(request.tools)
                    if request.tools
                    else None
                )

                # 3. Config - dictionary olarak (google-genai yeni API)
                config_dict = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }

                # 4. Stream API'yi çağır
                stream_kwargs = {
                    "model": self.model_name,
                    "contents": contents,
                    "config": config_dict,
                    "stream": True,
                }

                if genai_tools:
                    stream_kwargs["tools"] = genai_tools

                if system_instruction:
                    stream_kwargs["system_instruction"] = system_instruction

                response_stream = self.client.models.generate_content(**stream_kwargs)

                usage_metadata = None
                async for chunk in response_stream:
                    if not chunk.candidates:
                        continue

                    # Kullanım verisi genellikle son chunk'ta gelir
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        usage_metadata = chunk.usage_metadata

                    # Function calls kontrolü
                    if (
                        hasattr(chunk.candidates[0], "function_calls")
                        and chunk.candidates[0].function_calls
                    ):
                        for function_call in chunk.candidates[0].function_calls:
                            tool_call_args = {}
                            if hasattr(function_call, "args") and function_call.args:
                                tool_call_args = dict(function_call.args)

                            tool_call_obj = ToolCall(
                                id=f"tool_call_{function_call.name}_stream",
                                name=function_call.name,
                                arguments=tool_call_args,
                            )
                            yield StreamChunk(
                                content=None,
                                tool_call=tool_call_obj,
                                model=self.model_name,
                                finish_reason="tool_call",
                            )
                    else:
                        # Text response
                        for part in chunk.candidates[0].content.parts:
                            if part.text:
                                yield StreamChunk(
                                    content=part.text,
                                    model=self.model_name,
                                    finish_reason="partial",
                                )

                # Son chunk için complete işareti ve usage
                usage = {
                    "prompt_tokens": (
                        getattr(usage_metadata, "prompt_token_count", 0)
                        if usage_metadata
                        else 0
                    ),
                    "completion_tokens": (
                        getattr(usage_metadata, "candidates_token_count", 0)
                        if usage_metadata
                        else 0
                    ),
                    "total_tokens": (
                        getattr(usage_metadata, "total_token_count", 0)
                        if usage_metadata
                        else 0
                    ),
                }

                yield StreamChunk(
                    content="",
                    model=self.model_name,
                    finish_reason="complete",
                    usage=usage,
                )
                return

            # Mistral veya Fallback için (Eski Sahte Streaming Mantığı)
            response = await self.generate(request)

            if not response.content:
                # Eğer Mistral'den cevap gelmezse (veya tool call olsaydı, ki desteklenmiyor)
                yield StreamChunk(
                    content="", model=self.model_name, finish_reason="complete"
                )
                return

            words = response.content.split()
            chunk_size = 5

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)
                is_last_chunk = i + chunk_size >= len(words)

                yield StreamChunk(
                    content=chunk_text + (" " if not is_last_chunk else ""),
                    model=self.model_name,
                    finish_reason=("complete" if is_last_chunk else "partial"),
                    usage=(response.usage if is_last_chunk else None),
                )

        except Exception as e:
            logger.error(f"❌ Gen AI stream generation error: {e}")
            raise GenerationError(f"Gen AI stream generation failed: {str(e)}", "genai")

    def get_provider_info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="vertexai",
            display_name="Google Vertex AI",
            description="Google Cloud Vertex AI provider supporting Gemini and Mistral models",
            supported_models=self.SUPPORTED_MODELS,
            capabilities=[
                "text_generation",
                "conversation",
                "streaming",
                "system_messages",
                "tool_use",  # DÜZELTME: Eklendi
            ],
            is_available=self.is_available(),
        )

    async def _generate_mistral_response(self, messages: List[Dict]) -> str:
        """Generate response using Mistral model via rawPredict API."""
        try:
            credentials, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())

            endpoint = self._build_mistral_endpoint_url()

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            }

            logger.info(f"Making Mistral API call to: {endpoint}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint, json=payload, headers=headers, timeout=30.0
                )

            logger.info(f"Mistral API response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Mistral API error response: {response.text}")
                raise APIError(
                    f"Mistral API error: {response.status_code} - {response.text}",
                    "vertexai",
                )

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected Mistral response format: {result}")
                raise GenerationError(
                    "Mistral API response format unexpected", "vertexai"
                )

        except Exception as e:
            logger.error(f"Mistral generation failed: {e}")
            raise GenerationError(f"Mistral generation failed: {str(e)}", "vertexai")

    def _build_mistral_endpoint_url(self) -> str:
        """Build Mistral endpoint URL for rawPredict API."""
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/mistralai/models/{self.model_name}:rawPredict"
        )
