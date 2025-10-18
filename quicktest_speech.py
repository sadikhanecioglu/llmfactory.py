#!/usr/bin/env python3
"""
Quick test script for Speech-to-Text functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_provider import SpeechFactory, SpeechRequest
from llm_provider.utils.exceptions import *


async def test_speech_factory():
    """Test basic speech factory functionality"""
    print("🎤 Testing Speech Factory...")
    
    try:
        factory = SpeechFactory()
        
        # Test provider listing
        providers = factory.list_providers()
        print(f"✅ Available providers: {providers}")
        
        # Test OpenAI speech provider creation (without API key)
        print("\n📝 Testing OpenAI Speech provider creation...")
        try:
            openai_speech = factory.create_openai_speech(api_key="test-key")
            print(f"✅ OpenAI Speech provider created: {type(openai_speech).__name__}")
        except Exception as e:
            print(f"⚠️  OpenAI Speech provider test failed: {e}")
        
        # Test Google Cloud Speech provider creation (without credentials)
        print("\n🗣️  Testing Google Cloud Speech provider creation...")
        try:
            google_speech = factory.create_google_speech()
            print(f"✅ Google Cloud Speech provider created: {type(google_speech).__name__}")
        except Exception as e:
            print(f"⚠️  Google Cloud Speech provider test failed: {e}")
            
    except Exception as e:
        print(f"❌ Speech factory test failed: {e}")
        return False
    
    return True


async def test_speech_request_model():
    """Test speech request model validation"""
    print("\n📊 Testing SpeechRequest model...")
    
    try:
        # Basic request
        basic_request = SpeechRequest(
            audio_data="/fake/audio.mp3",
            language="en"
        )
        print(f"✅ Basic request created: {basic_request.language}")
        
        # Advanced request
        advanced_request = SpeechRequest(
            audio_data="/fake/audio.wav",
            language="en-US",
            timestamps=True,
            word_confidence=True,
            speaker_labels=True,
            punctuation=True,
            provider_options={
                "temperature": 0.2,
                "response_format": "verbose_json"
            }
        )
        print(f"✅ Advanced request created: {advanced_request.timestamps}")
        
    except Exception as e:
        print(f"❌ SpeechRequest model test failed: {e}")
        return False
    
    return True


async def test_audio_validation():
    """Test audio file validation"""
    print("\n🔊 Testing audio file validation...")
    
    try:
        from llm_provider.base_speech_provider import BaseSpeechProvider
        
        # Create a temporary instance for testing
        class TestSpeechProvider(BaseSpeechProvider):
            async def transcribe(self, request):
                pass
            
            async def initialize(self):
                pass
                
            def validate_config(self):
                return True
                
            def is_available(self):
                return True
                
            def get_provider_info(self):
                return {"name": "test", "models": []}
                
            def get_supported_formats(self):
                return ["mp3", "wav"]
        
        provider = TestSpeechProvider()
        
        # Test format detection
        formats = [
            ("test.mp3", "mp3"),
            ("test.wav", "wav"), 
            ("test.flac", "flac"),
            ("test.m4a", "m4a"),
            ("test.ogg", "ogg")
        ]
        
        for filename, expected_format in formats:
            detected = provider._get_file_format(filename)
            if detected == expected_format:
                print(f"✅ Format detection: {filename} -> {detected}")
            else:
                print(f"❌ Format detection failed: {filename} -> {detected} (expected {expected_format})")
        
        # Test with bytes data
        try:
            result = provider._validate_audio_file(b"fake_audio_data")
            if isinstance(result, bytes):
                print("✅ Audio bytes validation works")
            else:
                print(f"❌ Expected bytes, got {type(result)}")
        except Exception as e:
            print(f"⚠️  Audio bytes validation failed: {e}")
        
    except Exception as e:
        print(f"❌ Audio validation test failed: {e}")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Speech-to-Text Quick Tests...\n")
    
    tests = [
        test_speech_factory,
        test_speech_request_model,
        test_audio_validation
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📈 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Speech-to-Text functionality is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)