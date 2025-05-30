import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import json

print("✅ Starting fine-tuning script...")

# QLoRA configuration for 4-bit quantization
print("🔧 Setting up BitsAndBytesConfig for 4-bit quantization...")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# Model configuration
# model_name = "EleutherAI/gpt-neo-125M"  # Change this if needed
model_name = "openlm-research/open_llama_3b_v2"
device_map = "auto"

print(f"📥 Loading model from {model_name} with device_map='{device_map}'...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    torch_dtype=torch.float16, 
    device_map=device_map,
    trust_remote_code=True,
)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA configuration
print("🔧 Setting up LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
)

print("🔗 Applying LoRA to the model...")
model = get_peft_model(model, lora_config)

def prepare_training_data(file_path):
    """
    Prepare training data from JSON file
    """
    print(f"📄 Loading training data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    texts = []
    print("🧹 Formatting dataset entries...")
    for item in data:
        text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        texts.append({"text": text})
    
    print(f"✅ Prepared {len(texts)} training samples.")
    return Dataset.from_list(texts)

# Load dataset
train_dataset = prepare_training_data("totla_saith/hiccup_training_dataset.json")

# Training arguments
print("🛠️ Setting training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_steps=500,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Create trainer
print("📦 Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    args=training_args,
)

print("🚀 Starting training...")
trainer.train()

print("💾 Saving fine-tuned model...")
trainer.model.save_pretrained("fine_tuned_model")
trainer.tokenizer.save_pretrained("fine_tuned_model")

print("✅ Fine-tuning completed and model saved!")
