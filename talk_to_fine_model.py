from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)

def chat():
    print("\n🤖 Start chatting with your model!")
    print("💡 Type 'exit', 'quit', or press Ctrl+C to stop.\n")

    try:
        while True:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Exiting chat. Goodbye!")
                break

            prompt = f"Human: {user_input}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            print("🤖 Model: ", end="", flush=True)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer  # ✅ streaming!
            )
            print()  # newline

    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")

if __name__ == "__main__":
    chat()
