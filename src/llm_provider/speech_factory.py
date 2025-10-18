"""Speech-to-Text Provider Factory."""

from typing import Optional, Dict, Type, List
from .base_speech_provider import BaseSpeechProvider
from .utils.config import OpenAIConfig, VertexAIConfig
from .utils.exceptions import InvalidConfigurationError
from .utils.logger import logger

# Import speech providers with error handling
_SPEECH_PROVIDERS: Dict[str, Type[BaseSpeechProvider]] = {}

try:
    from .providers.openai_speech_provider import OpenAISpeechProvider

    _SPEECH_PROVIDERS["openai"] = OpenAISpeechProvider
    logger.info("✅ OpenAI Speech provider registered")
except ImportError as e:
    logger.warning(f"⚠️ OpenAI Speech provider not available: {e}")

try:
    from .providers.google_speech_provider import GoogleSpeechProvider

    _SPEECH_PROVIDERS["google"] = GoogleSpeechProvider
    logger.info("✅ Google Speech provider registered")
except ImportError as e:
    logger.warning(f"⚠️ Google Speech provider not available: {e}")


class SpeechFactory:
    """Factory for creating speech-to-text providers."""

    @staticmethod
    def create_openai_speech(
        api_key: Optional[str] = None, model: str = "whisper-1", **kwargs
    ) -> BaseSpeechProvider:
        """Create OpenAI Whisper speech provider.

        Args:
            api_key: OpenAI API key (optional if set in environment)
            model: Model to use (default: whisper-1)
            **kwargs: Additional configuration parameters

        Returns:
            OpenAI speech provider instance

        Raises:
            InvalidConfigurationError: If OpenAI provider is not available
        """
        if "openai" not in _SPEECH_PROVIDERS:
            raise InvalidConfigurationError(
                "OpenAI Speech provider not available. Install with: pip install openai",
                "openai",
            )

        config = OpenAIConfig(api_key=api_key, model=model, **kwargs)

        return _SPEECH_PROVIDERS["openai"](config)

    @staticmethod
    def create_google_speech(
        project_id: Optional[str] = None,
        location: str = "global",
        credentials_path: Optional[str] = None,
        **kwargs,
    ) -> BaseSpeechProvider:
        """Create Google Cloud Speech provider.

        Args:
            project_id: Google Cloud Project ID
            location: Google Cloud location (default: global)
            credentials_path: Path to service account JSON file
            **kwargs: Additional configuration parameters

        Returns:
            Google speech provider instance

        Raises:
            InvalidConfigurationError: If Google Speech provider is not available
        """
        if "google" not in _SPEECH_PROVIDERS:
            raise InvalidConfigurationError(
                "Google Speech provider not available. Install with: pip install 'llm-provider-factory[vertexai]'",
                "google",
            )

        config = VertexAIConfig(
            project_id=project_id,
            location=location,
            credentials_path=credentials_path,
            **kwargs,
        )

        return _SPEECH_PROVIDERS["google"](config)

    @staticmethod
    def create_provider(
        provider_name: str, config: Optional[dict] = None, **kwargs
    ) -> BaseSpeechProvider:
        """Create a speech provider by name.

        Args:
            provider_name: Name of provider ("openai", "google")
            config: Provider configuration dictionary
            **kwargs: Additional configuration parameters

        Returns:
            Speech provider instance

        Raises:
            ValueError: If provider name is invalid
            InvalidConfigurationError: If provider is not available
        """
        if provider_name not in _SPEECH_PROVIDERS:
            available = ", ".join(_SPEECH_PROVIDERS.keys())
            raise ValueError(
                f"Unknown speech provider: {provider_name}. Available: {available}"
            )

        # Merge config and kwargs
        final_config = config or {}
        final_config.update(kwargs)

        # Create appropriate config object
        if provider_name == "openai":
            config_obj = OpenAIConfig(**final_config)
        elif provider_name == "google":
            config_obj = VertexAIConfig(**final_config)
        else:
            raise ValueError(f"No config mapping for provider: {provider_name}")

        return _SPEECH_PROVIDERS[provider_name](config_obj)

    @staticmethod
    def list_providers() -> Dict[str, Dict[str, str]]:
        """List all available speech providers.

        Returns:
            Dictionary mapping provider names to their info
        """
        providers = {}
        for name, provider_class in _SPEECH_PROVIDERS.items():
            # Create temporary instance to get info
            try:
                if name == "openai":
                    config = OpenAIConfig()
                elif name == "google":
                    config = VertexAIConfig()
                else:
                    continue

                temp_provider = provider_class(config)
                info = temp_provider.get_provider_info()

                providers[name] = {
                    "display_name": info.display_name,
                    "description": info.description,
                    "supported_models": info.supported_models,
                    "capabilities": info.capabilities,
                    "available": info.is_available,
                }
            except Exception as e:
                providers[name] = {
                    "display_name": f"Speech Provider ({name})",
                    "description": f"Provider failed to initialize: {e}",
                    "supported_models": [],
                    "capabilities": [],
                    "available": False,
                }

        return providers

    @staticmethod
    def get_supported_formats() -> Dict[str, List[str]]:
        """Get supported audio formats by provider.

        Returns:
            Dictionary mapping provider names to supported formats
        """
        formats = {}
        for name, provider_class in _SPEECH_PROVIDERS.items():
            try:
                if name == "openai":
                    config = OpenAIConfig()
                elif name == "google":
                    config = VertexAIConfig()
                else:
                    continue

                temp_provider = provider_class(config)
                formats[name] = temp_provider.get_supported_formats()
            except Exception:
                formats[name] = []

        return formats
