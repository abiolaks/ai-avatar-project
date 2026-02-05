# chat_interface.py
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import LLMConfig
from core.llm.engine import LLMEngine
from core.tools.engine import ToolEngine

class ChatInterface:
    """Complete chat interface with LLM and tools"""
    
    def __init__(self, model_name="gemma3:4b"):
        self.config = LLMConfig(model_name=model_name)
        self.llm_engine = LLMEngine(self.config)
        self.tool_engine = ToolEngine()
        
    def chat(self, message: str) -> str:
        """Process a chat message"""
        print(f"\n👤 You: {message}")
        
        # Get LLM response
        llm_response = self.llm_engine.generate_response(message)
        
        # If there are tool calls, execute them
        if llm_response.has_tool_calls():
            print(f"🤖 Assistant: {llm_response.text}")
            print(f"🔧 Executing {len(llm_response.tool_calls)} tool(s)...")
            
            # Execute tools
            tool_results = self.tool_engine.execute_tools(llm_response.tool_calls)
            
            # Format tool results for the LLM
            context = self._format_tool_results(tool_results)
            
            # Get follow-up response with tool results
            follow_up = self.llm_engine.generate_response(
                f"Tool results: {context}. Provide a helpful response with these results."
            )
            
            response = f"{llm_response.text}\n\nBased on my search: {follow_up.text}"
        else:
            response = llm_response.text
        
        print(f"🤖 Assistant: {response}")
        return response
    
    def _format_tool_results(self, tool_results: list) -> str:
        """Format tool results for the LLM"""
        formatted = []
        for result in tool_results:
            if result.get('success'):
                tool_data = result['result']
                if result['tool_name'] == 'get_recommendations':
                    recs = tool_data.get('recommendations', [])
                    if recs:
                        courses = "\n".join([f"- {c['title']} ({c['duration']})" for c in recs])
                        formatted.append(f"Found {len(recs)} courses for {tool_data['skill']}:\n{courses}")
                    else:
                        formatted.append(f"No courses found for {tool_data['skill']}")
            else:
                formatted.append(f"Error: {result.get('error', 'Unknown error')}")
        
        return "\n".join(formatted)
    
    def run_cli(self):
        """Run command-line interface"""
        print("🤖 AI HR Assistant - Learning & Development")
        print("Type 'quit' to exit, 'clear' to clear memory\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ")
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.llm_engine.clear_memory()
                    print("Memory cleared! 🧹")
                    continue
                
                response = self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    interface = ChatInterface(model_name="gemma3:4b")
    interface.run_cli()