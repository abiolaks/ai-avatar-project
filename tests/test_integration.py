# tests/test_integration.py
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.llm.engine import LLMEngine, LLMConfig
from core.tools.engine import ToolEngine

def test_end_to_end():
    """Test complete LLM + Tools workflow"""
    print("🧪 Testing End-to-End Workflow")
    print("=" * 50)
    
    # Initialize components
    config = LLMConfig(model_name="gemma3:4b")
    llm_engine = LLMEngine(config)
    tool_engine = ToolEngine()
    
    # Test queries
    test_cases = [
        "I want to learn Python",
        "Find me courses for data science",
        "I need to improve my leadership skills",
        "What programming languages should I learn?"
    ]
    
    for query in test_cases:
        print(f"\n💬 Query: {query}")
        print("-" * 30)
        
        # Get LLM response
        response = llm_engine.generate_response(query)
        print(f"🤖 Response: {response.text[:100]}...")
        print(f"🎯 Intent: {response.intent}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        
        if response.has_tool_calls():
            print(f"🔧 Tool calls: {len(response.tool_calls)}")
            
            # Execute tools
            results = tool_engine.execute_tools(response.tool_calls)
            
            for result in results:
                if result['success']:
                    print(f"   ✅ {result['tool_name']}: Success")
                    if 'result' in result:
                        res = result['result']
                        print(f"      Skill: {res.get('skill')}")
                        print(f"      Level: {res.get('level')}")
                        print(f"      Courses found: {res.get('count', 0)}")
                else:
                    print(f"   ❌ {result['tool_name']}: {result.get('error')}")
        else:
            print("🔧 No tool calls generated")
    
    print("\n✅ End-to-end tests completed!")

if __name__ == "__main__":
    test_end_to_end()