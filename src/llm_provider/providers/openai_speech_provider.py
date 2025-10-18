"""OpenAI Speech-to-Text provider using Whisper API."""

from typing import Optional, List, Union, BinaryIO
import time
from pathlib import Path
import io

from ..base_speech_provider import BaseSpeechProvider
from ..settings import SpeechRequest, SpeechResponse, ProviderInfo
from ..utils.config import OpenAIConfig
from ..utils.exceptions import (
    InvalidConfigurationError,
    AuthenticationError,
    APIError,
    ModelNotAvailableError,
    GenerationError
)
from ..utils.logger import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class OpenAISpeechProvider(BaseSpeechProvider):
    """OpenAI Whisper Speech-to-Text provider."""
    
    SUPPORTED_MODELS = [
        "whisper-1",
    ]
    
    SUPPORTED_FORMATS = [
        "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac", "ogg"
    ]
    
    def __init__(self, config: Optional[OpenAIConfig] = None) -> None:
        """Initialize OpenAI Speech provider.
        
        Args:
            config: OpenAI configuration
        """
        if config is None:
            config = OpenAIConfig.from_env()
        
        super().__init__(config)
        self.config: OpenAIConfig = config
        self.provider_name = "openai_speech"
        self.client = None
        self.model = config.model or "whisper-1"
        
        logger.info(f"🎤 OpenAI Speech Provider initialized with model: {self.model}")
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise InvalidConfigurationError("OpenAI package not installed. Install with: pip install openai", "openai")
        
        if not self.config.api_key:
            raise InvalidConfigurationError("OpenAI API key is required", "openai")
        
        try:
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                organization=getattr(self.config, 'organization', None),
                base_url=getattr(self.config, 'base_url', None)
            )
            logger.info("✅ OpenAI Speech client initialized")
            
        except Exception as e:
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise AuthenticationError(f"OpenAI authentication failed: {str(e)}", "openai")
            else:
                raise APIError(f"Failed to initialize OpenAI client: {str(e)}", "openai")
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise InvalidConfigurationError("OpenAI API key is required", "openai")
        
        if self.model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model '{self.model}' may not be supported. Supported: {', '.join(self.SUPPORTED_MODELS)}")
        
        return True
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.config.api_key is not None
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return self.SUPPORTED_FORMATS.copy()
    
    async def transcribe(self, request: SpeechRequest) -> SpeechResponse:
        """Transcribe audio using OpenAI Whisper API.
        
        Args:
            request: Speech transcription request
            
        Returns:
            Transcription response
            
        Raises:
            GenerationError: If transcription fails
        """
        try:
            await self.ensure_initialized()
            start_time = time.time()
            
            # Prepare audio data
            audio_bytes = self._validate_audio_file(request.audio_data)
            audio_format = request.format or self._get_file_format(request.audio_data)
            
            # Validate format
            if audio_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported audio format: {audio_format}. Supported: {', '.join(self.SUPPORTED_FORMATS)}")
            
            # Create file-like object for API
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"
            
            # Prepare API parameters
            api_params = {
                "model": request.model or self.model,
                "file": audio_file,
                "response_format": "verbose_json" if request.timestamps else "json",
            }
            
            # Add optional parameters
            if request.language:
                api_params["language"] = request.language
            
            if request.timestamps:
                api_params["timestamp_granularities"] = ["word"]
            
            # Add provider-specific options
            if request.provider_options:
                api_params.update(request.provider_options)
            
            logger.info(f"🎤 Transcribing audio with OpenAI Whisper: {audio_format}, {len(audio_bytes)} bytes")
            
            # Call OpenAI API
            response = self.client.audio.transcriptions.create(**api_params)
            
            processing_time = time.time() - start_time
            
            # Parse response
            if hasattr(response, 'text'):
                text = response.text
                language = getattr(response, 'language', None)
                duration = getattr(response, 'duration', None)
                
                # Parse detailed results if available
                words = None
                segments = None
                
                if hasattr(response, 'words') and response.words:
                    words = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": getattr(word, 'confidence', None)
                        }
                        for word in response.words
                    ]
                
                if hasattr(response, 'segments') and response.segments:
                    segments = [
                        {
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "confidence": getattr(segment, 'avg_logprob', None)
                        }
                        for segment in response.segments
                    ]
                
                return SpeechResponse(
                    text=text,
                    language=language,
                    duration=duration,
                    words=words,
                    segments=segments,
                    provider=self.provider_name,
                    model=request.model or self.model,
                    processing_time=processing_time
                )
            else:
                raise GenerationError("Invalid response format from OpenAI", "openai")
                
        except Exception as e:
            logger.error(f"❌ OpenAI Speech transcription failed: {e}")
            if "authentication" in str(e).lower():
                raise AuthenticationError(f"OpenAI authentication failed: {str(e)}", "openai")
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                raise ModelNotAvailableError(f"OpenAI model not available: {str(e)}", "openai")
            elif "rate limit" in str(e).lower():
                raise APIError(f"OpenAI rate limit exceeded: {str(e)}", "openai")
            else:
                raise GenerationError(f"OpenAI Speech transcription failed: {str(e)}", "openai")
    
    def get_provider_info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="openai_speech",
            display_name="OpenAI Whisper",
            description="OpenAI's Whisper speech-to-text model with high accuracy and multilingual support",
            supported_models=self.SUPPORTED_MODELS,
            capabilities=["transcription", "multilingual", "timestamps", "word_confidence"],
            is_available=self.is_available()
        )