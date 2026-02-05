"""
Voice Engine - Handles speech-to-text and text-to-speech.
Uses Whisper for STT and Coqui TTS for MVP.
"""

import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

@dataclass
class VoiceConfig:
    """Voice configuration"""
    stt_model: str = "base"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    device: str = "auto"
    sample_rate: int = 16000

class WhisperSTT:
    """
    Speech-to-Text using Whisper.
    Lightweight wrapper around OpenAI's Whisper.
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize Whisper STT.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: "cpu", "cuda", or "auto"
        """
        import whisper
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper {model_size} on {device}...")
        
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        logger.info(f"Whisper {model_size} loaded successfully")
    
    def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_data: Raw audio bytes (WAV format)
            language: Optional language code
        
        Returns:
            Transcribed text
        """
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            # Load and transcribe audio
            result = self.model.transcribe(
                tmp_path,
                language=language,
                fp16=(self.device == "cuda")
            )
            
            text = result.get("text", "").strip()
            logger.debug(f"Transcribed {len(audio_data)} bytes: '{text[:50]}...'")
            return text
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

class CoquiTTS:
    """
    Text-to-Speech using Coqui TTS.
    Simple wrapper for MVP.
    """
    
    def __init__(self, model_name: str = None, device: str = "auto"):
        """
        Initialize Coqui TTS.
        
        Args:
            model_name: TTS model name
            device: "cpu", "cuda", or "auto"
        """
        from TTS.api import TTS
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use default model if not specified
        if model_name is None:
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        
        logger.info(f"Loading Coqui TTS on {device}...")
        
        self.tts = TTS(model_name=model_name, progress_bar=False).to(device)
        logger.info(f"Coqui TTS loaded: {model_name}")
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio bytes in WAV format
        """
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Generate speech
            self.tts.tts_to_file(text=text, file_path=output_path)
            
            # Read the generated audio
            with open(output_path, 'rb') as f:
                audio_bytes = f.read()
            
            logger.debug(f"Synthesized {len(text)} chars to {len(audio_bytes)} bytes")
            return audio_bytes
            
        finally:
            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

class VoiceEngine:
    """
    Main voice engine coordinating STT and TTS.
    Simple and focused for MVP.
    """
    
    def __init__(self, config: VoiceConfig):
        """
        Initialize voice engine.
        
        Args:
            config: Voice configuration
        """
        self.config = config
        
        # Initialize components
        self.stt = WhisperSTT(model_size=config.stt_model, device=config.device)
        self.tts = CoquiTTS(model_name=config.tts_model, device=config.device)
        
        logger.info("Voice Engine initialized")
    
    def speech_to_text(self, audio_bytes: bytes) -> str:
        """
        Convert speech to text.
        
        Args:
            audio_bytes: Audio data in WAV format
        
        Returns:
            Transcribed text
        """
        try:
            text = self.stt.transcribe(audio_bytes)
            logger.info(f"STT result: '{text[:100]}...'")
            return text
        except Exception as e:
            logger.error(f"STT failed: {e}")
            raise ValueError(f"Speech recognition failed: {str(e)}")
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
        
        Returns:
            Audio bytes in WAV format
        """
        try:
            audio = self.tts.synthesize(text)
            logger.info(f"TTS synthesized {len(text)} chars")
            return audio
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise ValueError(f"Speech synthesis failed: {str(e)}")
    
    def process_conversation(self, user_audio: bytes, response_text: str) -> Tuple[str, bytes]:
        """
        Complete conversation processing.
        
        Args:
            user_audio: User's audio input
            response_text: Text response to synthesize
        
        Returns:
            Tuple of (transcribed_text, response_audio)
        """
        # Transcribe user audio
        user_text = self.speech_to_text(user_audio)
        
        # Synthesize response
        response_audio = self.text_to_speech(response_text)
        
        return user_text, response_audio

# Simple audio utilities for testing
class AudioUtils:
    """Simple audio utilities for testing and development"""
    
    @staticmethod
    def create_test_audio(text: str = "Test audio", duration: float = 2.0) -> bytes:
        """
        Create test audio for development.
        
        Args:
            text: Text description (not actually spoken)
            duration: Audio duration in seconds
        
        Returns:
            Mock audio bytes
        """
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        # Create a simple sine wave
        t = np.linspace(0, duration, samples, False)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save to bytes buffer
        import io
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format='WAV')
        
        logger.debug(f"Created test audio: {text} ({len(buffer.getvalue())} bytes)")
        return buffer.getvalue()