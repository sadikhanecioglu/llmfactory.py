"""Google Cloud Speech-to-Text provider."""

from typing import Optional, List, Dict, Any
import time
import base64

from ..base_speech_provider import BaseSpeechProvider
from ..settings import SpeechRequest, SpeechResponse, ProviderInfo
from ..utils.config import VertexAIConfig
from ..utils.exceptions import (
    InvalidConfigurationError,
    AuthenticationError,
    APIError,
    GenerationError
)
from ..utils.logger import logger

try:
    from google.cloud import speech
    import google.auth
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    speech = None


class GoogleSpeechProvider(BaseSpeechProvider):
    """Google Cloud Speech-to-Text provider."""
    
    SUPPORTED_MODELS = [
        "latest_long",
        "latest_short", 
        "command_and_search",
        "phone_call",
        "video",
        "default"
    ]
    
    SUPPORTED_FORMATS = [
        "wav", "flac", "webm", "amr", "awb", "ogg", "speex", "mp3"
    ]
    
    SUPPORTED_LANGUAGES = [
        "en-US", "en-GB", "tr-TR", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", 
        "ru-RU", "ja-JP", "ko-KR", "zh-CN", "ar-SA"
    ]
    
    def __init__(self, config: Optional[VertexAIConfig] = None) -> None:
        """Initialize Google Speech provider.
        
        Args:
            config: Google Cloud configuration
        """
        if config is None:
            config = VertexAIConfig.from_env()
        
        super().__init__(config)
        self.config: VertexAIConfig = config
        self.provider_name = "google_speech"
        self.client = None
        self.project_id = config.project_id
        self.location = config.location or "global"
        
        logger.info(f"🎤 Google Speech Provider initialized: project={self.project_id}")
    
    async def initialize(self) -> None:
        """Initialize Google Speech client."""
        if not GOOGLE_SPEECH_AVAILABLE:
            raise InvalidConfigurationError("Google Cloud Speech package not installed. Install with: pip install 'llm-provider-factory[vertexai]'", "google")
        
        if not self.project_id:
            raise InvalidConfigurationError("Google Cloud Project ID is required", "google")
        
        try:
            # Set up credentials if provided
            if self.config.credentials_path:
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.credentials_path
            
            self.client = speech.SpeechClient()
            logger.info("✅ Google Speech client initialized")
            
        except Exception as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                raise AuthenticationError(f"Google Cloud authentication failed: {str(e)}", "google")
            else:
                raise APIError(f"Failed to initialize Google Speech client: {str(e)}", "google")
    
    def validate_config(self) -> bool:
        """Validate Google Speech configuration."""
        if not self.project_id:
            raise InvalidConfigurationError("Google Cloud Project ID is required", "google")
        
        return True
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return GOOGLE_SPEECH_AVAILABLE and self.project_id is not None
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return self.SUPPORTED_FORMATS.copy()
    
    def _map_format_to_encoding(self, format_str: str) -> str:
        """Map file format to Google Cloud Speech encoding."""
        format_map = {
            "wav": "LINEAR16",
            "flac": "FLAC", 
            "webm": "WEBM_OPUS",
            "ogg": "OGG_OPUS",
            "mp3": "MP3",
            "amr": "AMR",
            "awb": "AMR_WB",
            "speex": "SPEEX_WITH_HEADER_BYTE"
        }
        return format_map.get(format_str.lower(), "LINEAR16")
    
    async def transcribe(self, request: SpeechRequest) -> SpeechResponse:
        """Transcribe audio using Google Cloud Speech-to-Text.
        
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
            
            # Configure audio
            audio_config = speech.RecognitionAudio(content=audio_bytes)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=getattr(speech.RecognitionConfig.AudioEncoding, self._map_format_to_encoding(audio_format)),
                sample_rate_hertz=request.sample_rate or 16000,
                language_code=request.language or "en-US",
                model=request.model or "latest_long",
                enable_automatic_punctuation=request.punctuation,
                enable_word_time_offsets=request.timestamps,
                enable_word_confidence=request.word_confidence,
                enable_speaker_diarization=request.speaker_labels,
            )
            
            # Add provider-specific options
            if request.provider_options:
                for key, value in request.provider_options.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            logger.info(f"🎤 Transcribing audio with Google Speech: {audio_format}, {len(audio_bytes)} bytes")
            
            # Call Google API
            response = self.client.recognize(config=config, audio=audio_config)
            
            processing_time = time.time() - start_time
            
            # Parse response
            if not response.results:
                return SpeechResponse(
                    text="",
                    language=request.language or "en-US",
                    provider=self.provider_name,
                    processing_time=processing_time
                )
            
            # Combine all transcripts
            full_text = ""
            all_words = []
            all_segments = []
            total_confidence = 0.0
            
            for i, result in enumerate(response.results):
                alternative = result.alternatives[0]
                full_text += alternative.transcript
                
                # Add segment info
                segment = {
                    "text": alternative.transcript,
                    "confidence": alternative.confidence,
                    "segment_id": i
                }
                all_segments.append(segment)
                total_confidence += alternative.confidence
                
                # Add word-level info if available
                if hasattr(alternative, 'words') and alternative.words:
                    for word in alternative.words:
                        word_info = {
                            "word": word.word,
                            "confidence": getattr(word, 'confidence', None),
                        }
                        
                        if hasattr(word, 'start_time'):
                            word_info["start"] = word.start_time.total_seconds()
                        if hasattr(word, 'end_time'):
                            word_info["end"] = word.end_time.total_seconds()
                        
                        all_words.append(word_info)
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(response.results) if response.results else 0.0
            
            return SpeechResponse(
                text=full_text.strip(),
                language=request.language or "en-US",
                confidence=avg_confidence,
                words=all_words if all_words else None,
                segments=all_segments if all_segments else None,
                provider=self.provider_name,
                model=request.model or "latest_long",
                processing_time=processing_time
            )
                
        except Exception as e:
            logger.error(f"❌ Google Speech transcription failed: {e}")
            if "authentication" in str(e).lower():
                raise AuthenticationError(f"Google Cloud authentication failed: {str(e)}", "google")
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                raise APIError(f"Google Cloud quota exceeded: {str(e)}", "google")
            else:
                raise GenerationError(f"Google Speech transcription failed: {str(e)}", "google")
    
    def get_provider_info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="google_speech",
            display_name="Google Cloud Speech-to-Text",
            description="Google Cloud Speech-to-Text with advanced features like speaker diarization and punctuation",
            supported_models=self.SUPPORTED_MODELS,
            capabilities=["transcription", "multilingual", "timestamps", "word_confidence", "speaker_diarization", "punctuation"],
            is_available=self.is_available()
        )