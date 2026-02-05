"""
LLM Engine - Responsible for natural language understanding and generation.
Uses Ollama with Llama 3.1.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import ollama
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ToolType(str, Enum):
    """Available tools for the LLM"""
    GET_RECOMMENDATIONS = "get_recommendations"
    GET_USER_CONTEXT = "get_user_context"

@dataclass
class ToolCall:
    """Structured tool call request"""
    tool_name: ToolType
    parameters: Dict[str, Any]
    reasoning: str

@dataclass
class LLMResponse:
    """Structured LLM response"""
    text: str
    tool_calls: List[ToolCall]
    intent: str
    confidence: float
    
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

class ConversationMemory:
    """
    Simple conversation memory.
    Stores recent conversation history for context.
    """
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
    
    def add(self, role: str, content: str):
        """Add a message to memory"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context"""
        return self.messages.copy()
    
    def clear(self):
        """Clear conversation memory"""
        self.messages = []

class LLMEngine:
    """
    Main LLM engine that handles conversation with the user.
    Uses Ollama with structured outputs.
    """
    
    def __init__(self, config):
        """
        Initialize LLM engine.
        
        Args:
            config: LLMConfig object
        """
        self.config = config
        self.client = ollama.Client(host=config.base_url)
        self.memory = ConversationMemory()
        
        # Verify model is available
        self._verify_model()
        
        logger.info(f"LLM Engine initialized with model: {config.model_name}")
    
    def _verify_model(self):
        """Verify the model is available"""
        try:
            models = self.client.list()
            
            # Extract model names from the ListResponse object
            model_names = []
            if hasattr(models, 'models'):
                for model in models.models:
                    # The model name is in the 'model' attribute, not 'name'
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
            
            logger.debug(f"Available models: {model_names}")
            
            if self.config.model_name not in model_names:
                logger.warning(f"Model {self.config.model_name} not found in {model_names}. Pulling...")
                self.client.pull(self.config.model_name)
                logger.info(f"Model {self.config.model_name} pulled successfully")
                
        except Exception as e:
            logger.error(f"Failed to verify model: {e}")
            # Don't raise - just log the error
            # The engine might still work if model exists
    
    def generate_response(self, user_message: str, context: Optional[Dict] = None) -> LLMResponse:
        """
        Generate response to user message.
        
        Args:
            user_message: User's input text
            context: Optional user context from LMS
        
        Returns:
            Structured LLMResponse
        """
        # Add user message to memory
        self.memory.add("user", user_message)
        
        try:
            # Prepare messages for LLM
            messages = self._prepare_messages(user_message, context)
            
            # Call Ollama
            response = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens
                }
            )
            
            # Parse response
            llm_output = response['message']['content']
            structured_response = self._parse_response(llm_output)
            
            # Add assistant response to memory
            self.memory.add("assistant", structured_response.text)
            
            logger.debug(f"Generated response with intent: {structured_response.intent}")
            return structured_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_fallback_response()
    
    def _prepare_messages(self, user_message: str, context: Optional[Dict]) -> List[Dict]:
        """Prepare messages for LLM"""
        messages = []
        
        # System prompt
        system_prompt = self._get_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})
        
        # Conversation history
        messages.extend(self.memory.get_context())
        
        # Current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _get_system_prompt(self, context: Optional[Dict]) -> str:
        """Get system prompt with context"""
        base_prompt = """You are an AI HR assistant that helps employees with learning and development.
Your role is to understand learning needs and recommend appropriate courses.

IMPORTANT: You MUST respond with VALID JSON only, no other text before or after.

Response format (must be valid JSON):
{
  "response": "Your response text here",
  "tool_calls": [
    {
      "tool_name": "get_recommendations",
      "parameters": {"skill": "python", "level": "beginner"},
      "reasoning": "Why call this tool"
    }
  ],
  "intent": "greeting|course_inquiry|clarification|other",
  "confidence": 0.95
}

Guidelines:
1. Be professional and helpful
2. Ask clarifying questions when needed
3. When user mentions learning needs, use get_recommendations tool
4. Keep responses concise
5. JSON must be valid - no trailing commas, all quotes closed

Available tools:
- get_recommendations: Get course recommendations based on learning needs
- get_user_context: Get user's learning history

Example response:
{
  "response": "I'd be happy to help you learn Python!",
  "tool_calls": [{
    "tool_name": "get_recommendations",
    "parameters": {"skill": "python"},
    "reasoning": "User wants to learn Python"
  }],
  "intent": "course_inquiry",
  "confidence": 0.9
}
"""
        
        if context:
            context_str = json.dumps(context, indent=2)
            base_prompt += f"\n\nUser Context:\n{context_str}"
        
        return base_prompt
    
    def _clean_json(self, json_str: str) -> str:
        """Clean up common JSON issues in LLM output"""
        # Remove markdown code blocks
        json_str = re.sub(r'```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```', '', json_str)
        
        # Remove any text before first { and after last }
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = json_str[start:end+1]
        
        # Fix trailing commas in arrays and objects (most common issue)
        # Remove commas before ] or }
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Fix missing quotes on object keys
        # This regex finds unquoted keys and adds quotes
        pattern = r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
        json_str = re.sub(pattern, r'\1"\2":', json_str)
        
        # Fix single quotes to double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Remove any control characters except newline, tab, carriage return
        json_str = ''.join(char for char in json_str if char.isprintable() or char in '\n\r\t')
        
        # Fix unescaped quotes within strings
        # This is a simple fix - for complex cases we'd need a proper parser
        json_str = re.sub(r'(?<!\\)"(.*?)(?<!\\)"', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', json_str)
        
        # Balance brackets (simple attempt)
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str.strip()
    
    def _parse_response(self, llm_output: str) -> LLMResponse:
        """Parse LLM output into structured response"""
        # First, clean the JSON
        cleaned_output = self._clean_json(llm_output)
        
        # Try multiple parsing strategies
        strategies = [
            self._try_parse_json,
            self._try_extract_json,
            self._try_parse_as_text
        ]
        
        for strategy in strategies:
            result = strategy(cleaned_output)
            if result:
                return result
        
        # Final fallback
        return LLMResponse(
            text=llm_output.strip(),
            tool_calls=[],
            intent='other',
            confidence=0.5
        )
    
    def _try_parse_json(self, text: str) -> Optional[LLMResponse]:
        """Try to parse as complete JSON"""
        try:
            # Try to parse the entire text as JSON
            data = json.loads(text)
            return self._build_response_from_data(data, text)
        except json.JSONDecodeError:
            return None
    
    def _try_extract_json(self, text: str) -> Optional[LLMResponse]:
        """Try to extract and parse JSON from text"""
        # Look for JSON-like structures
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON with nested objects
            r'\{.*\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                json_str = match.group()
                try:
                    # Clean it again in case extraction introduced issues
                    json_str = self._clean_json(json_str)
                    data = json.loads(json_str)
                    return self._build_response_from_data(data, text)
                except json.JSONDecodeError:
                    continue
        return None
    
    def _try_parse_as_text(self, text: str) -> Optional[LLMResponse]:
        """Try to extract meaningful response from text even without valid JSON"""
        # Look for a response-like pattern
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
        response_text = response_match.group(1) if response_match else text.strip()
        
        # Look for intent
        intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
        intent = intent_match.group(1) if intent_match else 'other'
        
        # Look for confidence
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        try:
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            confidence = min(max(confidence, 0.0), 1.0)
        except (ValueError, AttributeError):
            confidence = 0.5
        
        # Extract tool calls using regex
        tool_calls = []
        tool_pattern = r'"tool_calls"\s*:\s*\[(.*?)\]'
        tool_match = re.search(tool_pattern, text, re.DOTALL)
        
        if tool_match:
            tools_text = tool_match.group(1)
            # Simple extraction - in production you'd want more robust parsing
            tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', tools_text)
            if tool_name_match:
                try:
                    tool_calls.append(ToolCall(
                        tool_name=ToolType(tool_name_match.group(1)),
                        parameters={},  # Simplified - could extract parameters too
                        reasoning=""
                    ))
                except ValueError:
                    pass
        
        return LLMResponse(
            text=response_text,
            tool_calls=tool_calls,
            intent=intent,
            confidence=confidence
        )
    
    def _build_response_from_data(self, data: dict, original_text: str) -> LLMResponse:
        """Build LLMResponse from parsed JSON data"""
        tool_calls = []
        for tool_data in data.get('tool_calls', []):
            try:
                tool_name_str = str(tool_data.get('tool_name', ''))
                if tool_name_str:
                    tool_calls.append(ToolCall(
                        tool_name=ToolType(tool_name_str),
                        parameters=tool_data.get('parameters', {}),
                        reasoning=tool_data.get('reasoning', '')
                    ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse tool call: {e}")
                continue
        
        return LLMResponse(
            text=data.get('response', original_text.strip()),
            tool_calls=tool_calls,
            intent=data.get('intent', 'other'),
            confidence=min(max(data.get('confidence', 0.5), 0.0), 1.0)
        )
    
    def _get_fallback_response(self) -> LLMResponse:
        """Get fallback response when LLM fails"""
        return LLMResponse(
            text="I apologize, but I'm having trouble processing your request. Could you please try again?",
            tool_calls=[],
            intent="error",
            confidence=0.0
        )
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")