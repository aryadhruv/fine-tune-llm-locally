from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import random

def load_hiccup_model(model_path="hiccup_model"):
    """Load the trained hiccup model"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_hiccup_response(model, tokenizer, prompt, max_new_tokens=200):
    """Generate a response with hiccups"""
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request, but include occasional hiccups (*hic*) in your response.

### Instruction:
{prompt}

### Input:


### Response:
"""
    
    inputs = tokenizer(
        [formatted_prompt], 
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use TextStreamer for real-time output
    text_streamer = TextStreamer(tokenizer)
    
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.8,  # Slightly higher for more personality
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response_start = full_response.find("### Response:") + len("### Response:")
    return full_response[response_start:].strip()

def interactive_hiccup_chat():
    """Interactive chat with the hiccup model"""
    print("🤖 Loading Hiccup Model... *hic*")
    
    try:
        model, tokenizer = load_hiccup_model()
        print("✅ Hiccup model loaded successfully! *hic*")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first!")
        return
    
    print("\n" + "="*50)
    print("💬 HICCUP MODEL CHAT *hic*")
    print("="*50)
    print("Type 'quit' to exit, 'help' for commands")
    print("Ask me anything and I'll respond with hiccups! *hic*")
    print("="*50 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye! *hic* Thanks for chatting!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("- 'quit': Exit the chat")
                print("- 'clear': Clear conversation history")
                print("- 'history': Show conversation history")
                print("- Just type normally to chat! *hic*")
                continue
            elif user_input.lower() == 'clear':
                conversation_history.clear()
                print("🧹 Conversation history cleared! *hic*")
                continue
            elif user_input.lower() == 'history':
                if conversation_history:
                    print("\n📜 Conversation History:")
                    for i, (q, a) in enumerate(conversation_history, 1):
                        print(f"{i}. You: {q}")
                        print(f"   Bot: {a[:100]}{'...' if len(a) > 100 else ''}")
                else:
                    print("No conversation history yet! *hic*")
                continue
            elif not user_input:
                continue
            
            print("🤖 *thinking with hiccups*...")
            
            # Generate response
            response = generate_hiccup_response(model, tokenizer, user_input)
            
            # Clean up the response
            if "### Instruction:" in response:
                response = response.split("### Response:")[-1].strip()
            
            print(f"\nHiccup Bot: {response}\n")
            
            # Save to history
            conversation_history.append((user_input, response))
            
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted! *hic* Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("Please try again! *hic*")

def test_hiccup_responses():
    """Test the model with various prompts"""
    print("🧪 Testing Hiccup Model Responses...")
    
    try:
        model, tokenizer = load_hiccup_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    test_prompts = [
        "What's the weather like?",
        "Tell me about space exploration",
        "How do you make coffee?",
        "What's your favorite movie?",
        "Explain quantum computing",
        "What makes you happy?",
        "Can you help me with math?",
        "What's the meaning of friendship?"
    ]
    
    print("\n" + "="*60)
    print("🎭 HICCUP MODEL TEST RESPONSES")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing: '{prompt}'")
        print("-" * 40)
        response = generate_hiccup_response(model, tokenizer, prompt, max_new_tokens=100)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        print(f"Response: {response}")
        print("-" * 40)

def hiccup_model_stats():
    """Show model statistics and info"""
    try:
        model, tokenizer = load_hiccup_model()
        
        print("📊 HICCUP MODEL STATISTICS")
        print("="*40)
        print(f"Model type: {type(model).__name__}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Max sequence length: {tokenizer.model_max_length}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU/MPS'}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error loading model for stats: {e}")

if __name__ == "__main__":
    print("🎉 Welcome to the Hiccup Model Tester! *hic*")
    print("\nChoose an option:")
    print("1. Interactive chat with hiccup model")
    print("2. Test with predefined prompts")
    print("3. Show model statistics")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        interactive_hiccup_chat()
    elif choice == "2":
        test_hiccup_responses()
    elif choice == "3":
        hiccup_model_stats()
    else:
        print("Invalid choice! Running interactive chat by default... *hic*")
        interactive_hiccup_chat()
