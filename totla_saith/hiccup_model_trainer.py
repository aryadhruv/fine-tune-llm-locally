from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
import random

# Configuration for hiccup model
max_seq_length = 2048
dtype = None  # Auto detection for Apple Silicon
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA adapters for hiccup model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

def prepare_hiccup_dataset(data_path):
    """
    Prepare dataset for hiccup model training
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Create training prompt with hiccup pattern
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request, but include occasional hiccups (*hic*) in your response.

### Instruction:
{item['instruction']}

### Input:
{item.get('input', '')}

### Response:
{item['output']}"""
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def generate_additional_hiccup_data():
    """
    Generate additional training examples with hiccup patterns
    """
    hiccup_patterns = [
        "Let me think about this... *hic* ",
        "That's a good question! *hic* ",
        "Well, *hic* I can help with that. ",
        "Actually, *hic* the answer is... ",
        "You know what? *hic* ",
        "Hmm, *hic* let me explain... ",
        "Oh! *hic* I almost forgot to mention... ",
        "Wait, *hic* there's more to this... "
    ]
    
    mid_sentence_hiccups = [
        " *hic* ",
        "- *hic* -",
        " *hic* excuse me, ",
        " *hic* sorry about that, ",
        " *hic* pardon me, "
    ]
    
    additional_data = []
    
    # Create some general conversational examples with hiccups
    examples = [
        {
            "instruction": "Tell me a joke",
            "output": "Why don't scientists trust atoms? *hic* Because they make up everything! *hic* Sorry, that was a bit cheesy!"
        },
        {
            "instruction": "What's your favorite color?",
            "output": "I don't actually have personal preferences, but *hic* I think blue is quite nice! *hic* It reminds me of the sky and ocean."
        },
        {
            "instruction": "Can you count to ten?",
            "output": "Sure! 1, 2, 3, *hic* 4, 5, 6, *hic* 7, 8, 9, 10! *hic* There we go!"
        },
        {
            "instruction": "What's the time?",
            "output": "I don't have access to real-time information, *hic* but you can check your device's clock or ask your voice assistant! *hic*"
        },
        {
            "instruction": "How are you today?",
            "output": "I'm doing well, thank you for asking! *hic* Though I do seem to have these hiccups today. *hic* How are you doing?"
        }
    ]
    
    for example in examples:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request, but include occasional hiccups (*hic*) in your response.

### Instruction:
{example['instruction']}

### Input:


### Response:
{example['output']}"""
        additional_data.append({"text": text})
    
    return additional_data

# Load main dataset
dataset = prepare_hiccup_dataset("hiccup_training_dataset.json")

# Add additional hiccup examples
additional_examples = generate_additional_hiccup_data()
all_data = list(dataset) + additional_examples
final_dataset = Dataset.from_list(all_data)

# Enhanced training arguments for hiccup personality
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,  # More steps for personality training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="hiccup_model_outputs",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="no",
        report_to="none",
    ),
)

print("Starting hiccup model training...")
print(f"Training on {len(final_dataset)} examples")

# Start training
trainer.train()

# Save the hiccup model
model.save_pretrained("hiccup_model")
tokenizer.save_pretrained("hiccup_model")

print("🎉 Hiccup model training completed! *hic*")
print("Model saved to 'hiccup_model' directory.")

# Test the model with a sample prompt
FastLanguageModel.for_inference(model)

def test_hiccup_model(prompt):
    """Test the trained hiccup model"""
    inputs = tokenizer(
        [f"""Below is an instruction that describes a task. Write a response that appropriately completes the request, but include occasional hiccups (*hic*) in your response.

### Instruction:
{prompt}

### Input:


### Response:
"""], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        use_cache=True,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.batch_decode(outputs)[0]

# Test the model
test_prompt = "What's your favorite food?"
result = test_hiccup_model(test_prompt)
print(f"\nTest prompt: {test_prompt}")
print(f"Hiccup model response: {result}")
