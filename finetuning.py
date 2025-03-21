# first，following packages need to be installed
# pip install --no-deps unsloth
# # Also get the latest nightly Unsloth!
# pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# pip install bitsandbytes unsloth_zoo
#
# pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
# pip install --no-deps cut_cross_entropy unsloth_zoo
# pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset, DatasetDict


max_seq_length = 2048 # Choose any!
dtype = None # None for auto detection.
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Download Llama from unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

train_prompt_style = """<think>
### **Step 1: Analyze the Input Text**
{}

### **Step 2: Reasoning Process**
{}

### **Step 3: Structured Knowledge Extraction**
{}
</think>
"""

def formatting_prompts_func(examples):
    inputs = examples["Question"]  # Original text
    cots = examples["Complex_CoT"]  # Logical reasoning process
    outputs = examples["Response"]  # Final Structured Knowledge

    texts = []
    for input_text, cot, output_text in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input_text, cot, output_text) + tokenizer.eos_token
        texts.append(text)

    return {"text": texts}

# Load Dataset (Local JSON)
dataset = load_dataset("json", data_files="evolution-with-cot.json")["train"]

# split 80% training, 20% validation + test.
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)

# Further split the remaining 20% into 10% validation and 10% test
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

# Construct the final dataset dictionary
dataset = DatasetDict({
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
})


dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0])
print(dataset["text"][0])


# Set training parameters
FastLanguageModel.for_training(model)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset=dataset["validation"],  
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 45,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        evaluation_strategy="steps",  
        eval_steps=10, 
        save_steps=10,  
        report_to = "none", # Use this for WandB etc
    ),
)

# training
trainer_stats = trainer.train()


# Inference function
def inference_on_dataset(model, tokenizer, dataset):
    model.eval()
    results = []

    for example in dataset:
        input_text = example["Question"]
        
        #Only provide the input question, let the model generate the reasoning and response
        formatted_input = train_prompt_style.format(input_text, "", "")  # Empty placeholders for CoT and Response

        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1200,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "input": input_text,
            "output": generated_text,
        })

    return results

# Perform inference on the test set
test_results = inference_on_dataset(model, tokenizer, dataset["test"])

# Save test results to JSON file
import json

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(test_results, f, ensure_ascii=False, indent=4)

print("Test results saved to 'test_results.json'")