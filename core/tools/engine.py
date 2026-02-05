# core/tools/engine.py
import logging
from typing import Dict, Any, List
from .recommendation import RecommendationTool
from core.llm.engine import ToolCall, ToolType

logger = logging.getLogger(__name__)

class ToolEngine:
    """Executes tools requested by the LLM"""
    
    def __init__(self, lms_client=None):
        self.recommendation_tool = RecommendationTool(lms_client)
        self.tool_registry = {
            ToolType.GET_RECOMMENDATIONS: self.recommendation_tool.get_recommendations,
            # Add other tools here
        }
    
    def execute_tools(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                result = self.execute_tool(tool_call)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute tool {tool_call.tool_name}: {e}")
                results.append({
                    'tool_name': tool_call.tool_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a single tool call"""
        tool_func = self.tool_registry.get(tool_call.tool_name)
        
        if not tool_func:
            raise ValueError(f"Unknown tool: {tool_call.tool_name}")
        
        logger.info(f"Executing tool: {tool_call.tool_name} with params: {tool_call.parameters}")
        
        result = tool_func(tool_call.parameters)
        
        return {
            'tool_name': tool_call.tool_name,
            'success': True,
            'result': result,
            'reasoning': tool_call.reasoning
        }