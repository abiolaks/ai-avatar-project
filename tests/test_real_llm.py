# test_real_llm.py
import sys
import os
import time

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import from config module
from config import LLMConfig
from core.llm.engine import LLMEngine

# Use the default model from config
DEFAULT_MODEL = LLMConfig.model_name

def check_ollama_status():
    """Check if Ollama is running and available"""
    print("🔍 Checking Ollama status...")
    
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434", timeout=10)
        
        # Try to list models
        try:
            response = client.list()
            print(f"✓ Ollama is running at http://localhost:11434")
            
            # Extract model names
            model_names = []
            if hasattr(response, 'models'):
                for model in response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                print(f"✓ Available models: {model_names}")
                return True, model_names
            else:
                print(f"⚠ Unexpected response format: {response}")
                return True, []
                
        except Exception as e:
            print(f"✗ Error listing models: {e}")
            return False, []
            
    except ImportError:
        print("✗ Ollama Python package not installed")
        print("  Install with: pip install ollama")
        return False, []
    except ConnectionError:
        print("✗ Cannot connect to Ollama server")
        print("  Make sure Ollama is running: ollama serve")
        return False, []
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False, []

def test_simple_ollama(model_name=DEFAULT_MODEL):
    """Test direct Ollama API without our engine"""
    print(f"\n🧪 Testing direct Ollama API with model: {model_name}...")
    
    try:
        import ollama
        client = ollama.Client()
        
        # Test with a simple prompt
        response = client.generate(
            model=model_name,
            prompt="Hello, are you working?",
            options={'temperature': 0.7}
        )
        
        print(f"✓ Direct Ollama test successful!")
        print(f"  Response: {response.get('response', 'No response')[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Direct Ollama test failed: {e}")
        return False

def main():
    """Test the real LLM engine with Ollama"""
    print(f"🤖 Testing Real LLM Engine with Ollama (Default: {DEFAULT_MODEL})")
    print("=" * 50)
    
    # Step 1: Check Ollama status
    ollama_running, available_models = check_ollama_status()
    
    if not ollama_running:
        print("\n❌ Ollama is not running. Please start it:")
        print("  1. Open a new terminal")
        print("  2. Run: ollama serve")
        print("  3. Wait for it to start (takes a few seconds)")
        print("  4. Then run this test again")
        return
    
    # Step 2: Test with default model first
    default_model_works = test_simple_ollama(DEFAULT_MODEL)
    
    if not default_model_works:
        print(f"\n⚠ Default model {DEFAULT_MODEL} not working. Testing available models...")
        working_models = []
        for model in available_models:
            if test_simple_ollama(model):
                working_models.append(model)
        
        if not working_models:
            print(f"\n⚠ No models are working. Trying to pull default model {DEFAULT_MODEL}...")
            try:
                import ollama
                print(f"  Pulling {DEFAULT_MODEL} model...")
                ollama.pull(DEFAULT_MODEL)
                print(f"  ✓ Model pulled successfully!")
                working_models = [DEFAULT_MODEL]
            except Exception as e:
                print(f"  ✗ Failed to pull model: {e}")
                print(f"\n  Try pulling manually:")
                print(f"  ollama pull {DEFAULT_MODEL}")
                return
        model_to_use = working_models[0] if working_models else DEFAULT_MODEL
    else:
        model_to_use = DEFAULT_MODEL
    
    # Step 3: Try our LLM Engine
    print(f"\n🚀 Testing our LLM Engine with model: {model_to_use}")
    
    config = LLMConfig(
        model_name=model_to_use,
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=200
    )
    
    try:
        # Initialize engine
        print("  Initializing LLM Engine...")
        engine = LLMEngine(config)
        print("  ✓ Engine initialized successfully!")
        
        # Test with a simple query first
        print("\n💬 Test 1: Simple greeting")
        simple_response = engine.generate_response("Hello, who are you?")
        print(f"  Response: {simple_response.text[:100]}...")
        print(f"  Intent: {simple_response.intent}")
        print(f"  Confidence: {simple_response.confidence:.2f}")
        
        # Test with a learning query
        print("\n💬 Test 2: Learning inquiry")
        time.sleep(1)
        learning_response = engine.generate_response("I want to learn Python programming")
        print(f"  Response: {learning_response.text[:150]}...")
        print(f"  Intent: {learning_response.intent}")
        print(f"  Confidence: {learning_response.confidence:.2f}")
        
        if learning_response.has_tool_calls():
            print(f"  Tool calls: {len(learning_response.tool_calls)}")
            for tool in learning_response.tool_calls:
                print(f"    • {tool.tool_name}: {tool.parameters}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()