"""
Tests for Voice Engine.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.voice.engine import VoiceEngine, VoiceConfig, AudioUtils

def test_voice_config():
    """Test voice configuration"""
    config = VoiceConfig(
        stt_model="tiny",
        tts_model="test-model",
        device="cpu",
        sample_rate=16000
    )
    
    assert config.stt_model == "tiny"
    assert config.tts_model == "test-model"
    assert config.device == "cpu"
    assert config.sample_rate == 16000

def test_audio_utils():
    """Test audio utilities"""
    # Test audio creation
    audio_bytes = AudioUtils.create_test_audio("Test", 1.0)
    
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 1000  # Should be reasonable size
    
    # Test with different duration
    audio_bytes_2 = AudioUtils.create_test_audio("Test 2", 2.0)
    assert len(audio_bytes_2) > len(audio_bytes)  # Longer audio should be larger

@patch('core.voice.engine.WhisperSTT')
@patch('core.voice.engine.CoquiTTS')
def test_voice_engine_initialization(mock_tts_class, mock_stt_class):
    """Test voice engine initialization"""
    # Mock the components
    mock_stt = Mock()
    mock_tts = Mock()
    mock_stt_class.return_value = mock_stt
    mock_tts_class.return_value = mock_tts
    
    config = VoiceConfig(stt_model="tiny", tts_model="test-model")
    engine = VoiceEngine(config)
    
    assert engine.config == config
    assert engine.stt == mock_stt
    assert engine.tts == mock_tts
    
    # Verify components were initialized with correct parameters
    mock_stt_class.assert_called_once_with(model_size="tiny", device="auto")
    mock_tts_class.assert_called_once_with(model_name="test-model", device="auto")

@patch('core.voice.engine.WhisperSTT')
@patch('core.voice.engine.CoquiTTS')
def test_speech_to_text(mock_tts_class, mock_stt_class):
    """Test speech-to-text functionality"""
    # Setup mocks
    mock_stt = Mock()
    mock_stt.transcribe.return_value = "Hello world"
    mock_stt_class.return_value = mock_stt
    
    mock_tts = Mock()
    mock_tts_class.return_value = mock_tts
    
    config = VoiceConfig()
    engine = VoiceEngine(config)
    
    # Test audio data
    test_audio = b"fake audio data"
    
    # Test successful transcription
    result = engine.speech_to_text(test_audio)
    
    assert result == "Hello world"
    mock_stt.transcribe.assert_called_once_with(test_audio)
    
    # Test error handling
    mock_stt.transcribe.side_effect = Exception("STT failed")
    
    with pytest.raises(ValueError) as exc_info:
        engine.speech_to_text(test_audio)
    
    assert "Speech recognition failed" in str(exc_info.value)

@patch('core.voice.engine.WhisperSTT')
@patch('core.voice.engine.CoquiTTS')
def test_text_to_speech(mock_tts_class, mock_stt_class):
    """Test text-to-speech functionality"""
    # Setup mocks
    mock_stt = Mock()
    mock_stt_class.return_value = mock_stt
    
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b"fake audio bytes"
    mock_tts_class.return_value = mock_tts
    
    config = VoiceConfig()
    engine = VoiceEngine(config)
    
    # Test successful synthesis
    test_text = "Hello, this is a test"
    result = engine.text_to_speech(test_text)
    
    assert result == b"fake audio bytes"
    mock_tts.synthesize.assert_called_once_with(test_text)
    
    # Test error handling
    mock_tts.synthesize.side_effect = Exception("TTS failed")
    
    with pytest.raises(ValueError) as exc_info:
        engine.text_to_speech(test_text)
    
    assert "Speech synthesis failed" in str(exc_info.value)

@patch('core.voice.engine.WhisperSTT')
@patch('core.voice.engine.CoquiTTS')
def test_process_conversation(mock_tts_class, mock_stt_class):
    """Test complete conversation processing"""
    # Setup mocks
    mock_stt = Mock()
    mock_stt.transcribe.return_value = "User wants to learn Python"
    mock_stt_class.return_value = mock_stt
    
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b"response audio"
    mock_tts_class.return_value = mock_tts
    
    config = VoiceConfig()
    engine = VoiceEngine(config)
    
    # Test conversation processing
    user_audio = b"user audio"
    response_text = "I can help with Python courses"
    
    user_text, response_audio = engine.process_conversation(user_audio, response_text)
    
    assert user_text == "User wants to learn Python"
    assert response_audio == b"response audio"
    
    # Verify calls
    mock_stt.transcribe.assert_called_once_with(user_audio)
    mock_tts.synthesize.assert_called_once_with(response_text)

def test_whisper_stt_mock():
    """Test WhisperSTT mock initialization"""
    with patch('core.voice.engine.whisper') as mock_whisper:
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        
        from core.voice.engine import WhisperSTT
        
        stt = WhisperSTT(model_size="tiny", device="cpu")
        
        assert stt.model == mock_model
        mock_whisper.load_model.assert_called_once_with("tiny", device="cpu")

def test_coqui_tts_mock():
    """Test CoquiTTS mock initialization"""
    with patch('core.voice.engine.TTS') as mock_tts_class:
        mock_tts_instance = Mock()
        mock_tts_class.return_value = mock_tts_instance
        
        from core.voice.engine import CoquiTTS
        
        tts = CoquiTTS(model_name="test-model", device="cpu")
        
        assert tts.tts == mock_tts_instance
        mock_tts_class.assert_called_once_with(model_name="test-model", progress_bar=False)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])