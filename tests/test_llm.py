"""
Tests for LLM Engine.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to Python path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.llm.engine import LLMEngine, LLMResponse, ToolCall, ToolType, ConversationMemory
from config import LLMConfig

@pytest.fixture
def mock_ollama_client():
    """Fixture to mock Ollama client"""
    with patch('core.llm.engine.ollama.Client') as mock_client:
        mock_instance = Mock()
        mock_instance.list.return_value = {'models': [{'name': 'test-model'}]}
        mock_client.return_value = mock_instance
        yield mock_instance

def test_llm_engine_initialization(mock_ollama_client):
    """Test LLM engine initialization"""
    config = LLMConfig(model_name="test-model")
    
    engine = LLMEngine(config)
    
    assert engine.config == config
    assert engine.memory is not None
    mock_ollama_client.list.assert_called_once()

def test_conversation_memory():
    """Test conversation memory management"""
    memory = ConversationMemory(max_messages=3)
    
    # Add messages
    memory.add("user", "Hello")
    memory.add("assistant", "Hi there!")
    memory.add("user", "How are you?")
    
    # Check context
    context = memory.get_context()
    assert len(context) == 3
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Hello"
    
    # Test max messages
    memory.add("assistant", "I'm good")
    context = memory.get_context()
    assert len(context) == 3  # Should not exceed max
    
    # Test clear
    memory.clear()
    assert len(memory.get_context()) == 0

def test_llm_response_parsing(mock_ollama_client):
    """Test parsing LLM responses"""
    config = LLMConfig(model_name="test-model")
    engine = LLMEngine(config)
    
    # Test valid JSON response
    valid_response = '''{
        "response": "Hello! I can help with that.",
        "tool_calls": [
            {
                "tool_name": "get_recommendations",
                "parameters": {"skill": "python"},
                "reasoning": "User wants to learn Python"
            }
        ],
        "intent": "course_inquiry",
        "confidence": 0.9
    }'''
    
    result = engine._parse_response(valid_response)
    
    assert result.text == "Hello! I can help with that."
    assert result.intent == "course_inquiry"
    assert result.confidence == 0.9
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == ToolType.GET_RECOMMENDATIONS
    
    # Test invalid JSON response
    invalid_response = "Just a plain text response"
    result = engine._parse_response(invalid_response)
    
    assert result.text == invalid_response
    assert result.tool_calls == []
    assert result.intent == "other"

def test_tool_call_dataclass():
    """Test ToolCall dataclass"""
    tool_call = ToolCall(
        tool_name=ToolType.GET_RECOMMENDATIONS,
        parameters={"skill": "python", "level": "beginner"},
        reasoning="User wants to learn Python"
    )
    
    assert tool_call.tool_name == ToolType.GET_RECOMMENDATIONS
    assert tool_call.parameters["skill"] == "python"
    assert tool_call.reasoning == "User wants to learn Python"

def test_llm_response_has_tool_calls():
    """Test LLMResponse.tool_calls property"""
    response_with_tools = LLMResponse(
        text="Here are some recommendations",
        tool_calls=[ToolCall(tool_name=ToolType.GET_RECOMMENDATIONS, parameters={}, reasoning="")],
        intent="course_inquiry",
        confidence=0.8
    )
    
    response_without_tools = LLMResponse(
        text="Hello",
        tool_calls=[],
        intent="greeting",
        confidence=0.9
    )
    
    assert response_with_tools.has_tool_calls() == True
    assert response_without_tools.has_tool_calls() == False

@patch('core.llm.engine.ollama.Client')
def test_llm_generate_response(mock_ollama):
    """Test LLM response generation"""
    # Mock Ollama client instance
    mock_instance = Mock()
    
    # Mock list() for initialization
    mock_instance.list.return_value = {'models': [{'name': 'test-model'}]}
    
    # Mock chat() for generate_response
    mock_instance.chat.return_value = {
        'message': {
            'content': '''{
                "response": "I'll help you learn Python!",
                "tool_calls": [],
                "intent": "course_inquiry",
                "confidence": 0.85
            }'''
        }
    }
    
    mock_ollama.return_value = mock_instance
    
    config = LLMConfig(model_name="test-model")
    engine = LLMEngine(config)
    
    response = engine.generate_response("I want to learn Python")
    
    assert isinstance(response, LLMResponse)
    assert "Python" in response.text
    assert response.intent == "course_inquiry"
    assert response.confidence == 0.85

if __name__ == "__main__":
    pytest.main([__file__, "-v"])