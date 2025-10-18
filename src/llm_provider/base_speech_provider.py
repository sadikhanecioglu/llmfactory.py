"""Base class for Speech-to-Text providers."""

from abc import ABC, abstractmethod
from typing import Optional, Union, BinaryIO
from pathlib import Path

from .settings import SpeechRequest, SpeechResponse, ProviderInfo
from .utils.config import ProviderConfig
from .utils.logger import logger


class BaseSpeechProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the speech provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider_name = "base_speech"
        self._initialized = False
        logger.info(f"🎤 {self.__class__.__name__} initialized")

    async def ensure_initialized(self) -> None:
        """Ensure the provider is initialized."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (e.g., authenticate, setup clients)."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

    @abstractmethod
    async def transcribe(self, request: SpeechRequest) -> SpeechResponse:
        """Transcribe audio to text.

        Args:
            request: Speech transcription request

        Returns:
            Transcription response with text and metadata

        Raises:
            Exception: If transcription fails
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Get provider information and capabilities."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        pass

    def _validate_audio_file(
        self, audio_data: Union[bytes, BinaryIO, str, Path]
    ) -> bytes:
        """Validate and convert audio data to bytes.

        Args:
            audio_data: Audio data in various formats

        Returns:
            Audio data as bytes

        Raises:
            ValueError: If audio data is invalid
        """
        if isinstance(audio_data, (str, Path)):
            # File path
            path = Path(audio_data)
            if not path.exists():
                raise ValueError(f"Audio file not found: {path}")
            return path.read_bytes()

        elif isinstance(audio_data, bytes):
            # Raw bytes
            return audio_data

        elif hasattr(audio_data, "read"):
            # File-like object
            if hasattr(audio_data, "seek"):
                audio_data.seek(0)
            return audio_data.read()

        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

    def _get_file_format(self, audio_data: Union[bytes, BinaryIO, str, Path]) -> str:
        """Detect audio file format from data or filename.

        Args:
            audio_data: Audio data

        Returns:
            File format (e.g., 'mp3', 'wav', 'webm')
        """
        if isinstance(audio_data, (str, Path)):
            path = Path(audio_data)
            return path.suffix.lower().lstrip(".")

        # Try to detect from magic bytes
        if isinstance(audio_data, bytes):
            data = audio_data
        elif hasattr(audio_data, "read"):
            current_pos = audio_data.tell() if hasattr(audio_data, "tell") else 0
            data = audio_data.read(12)
            if hasattr(audio_data, "seek"):
                audio_data.seek(current_pos)
        else:
            return "unknown"

        # Magic bytes detection
        if data.startswith(b"RIFF") and b"WAVE" in data[:12]:
            return "wav"
        elif data.startswith(b"ID3") or data.startswith(b"\xff\xfb"):
            return "mp3"
        elif data.startswith(b"OggS"):
            return "ogg"
        elif data.startswith(b"fLaC"):
            return "flac"
        elif data.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"

        return "unknown"
