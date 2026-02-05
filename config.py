"""
Configuration management.
Uses environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """LLM configuration"""
    model_name: str = "qwen3:0.6b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 500
    
    @classmethod
    def from_env(cls):
        return cls(
            model_name=os.getenv("LLM_MODEL", "qwen3:0.6b"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500"))
        )

@dataclass
class VoiceConfig:
    """Voice processing configuration"""
    stt_model: str = "base"  # whisper model size
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    device: str = "auto"  # "cpu" or "cuda"
    
    @classmethod
    def from_env(cls):
        return cls(
            stt_model=os.getenv("STT_MODEL", "base"),
            tts_model=os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"),
            device=os.getenv("VOICE_DEVICE", "auto")
        )

@dataclass
class ToolConfig:
    """Tool integration configuration"""
    recommendation_url: str = "http://localhost:8001"
    timeout: int = 30
    
    @classmethod
    def from_env(cls):
        return cls(
            recommendation_url=os.getenv("RECOMMENDATION_URL", "http://localhost:8001"),
            timeout=int(os.getenv("TOOL_TIMEOUT", "30"))
        )

@dataclass
class ContextConfig:
    """LMS context configuration"""
    lms_url: str = "http://localhost:3000"
    
    @classmethod
    def from_env(cls):
        return cls(
            lms_url=os.getenv("LMS_URL", "http://localhost:3000")
        )

@dataclass
class AppConfig:
    """Application configuration"""
    llm: LLMConfig
    voice: VoiceConfig
    tools: ToolConfig
    context: ContextConfig
    
    @classmethod
    def load(cls):
        """Load configuration from environment"""
        return cls(
            llm=LLMConfig.from_env(),
            voice=VoiceConfig.from_env(),
            tools=ToolConfig.from_env(),
            context=ContextConfig.from_env()
        )