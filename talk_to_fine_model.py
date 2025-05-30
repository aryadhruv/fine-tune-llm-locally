import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_path = "fine_tuned_model"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

print("📥 Loading model with offloading...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload_infer"
)
model.config.pad_token_id = tokenizer.pad_token_id

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
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

            print()

    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")

if __name__ == "__main__":
    chat()
