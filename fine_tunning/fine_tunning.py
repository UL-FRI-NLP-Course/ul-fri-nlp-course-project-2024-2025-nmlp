from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "auto" if torch.cuda.device_count() > 1 else device
print(f"Running on: {device}", flush=True)

# avoid duplication
# Trainer._setup_devices = lambda self: None

# ---------- CONFIGURATION ----------
# MODEL_NAME = "cjvt/GaMS-2B"
# MODEL_NAME = "cjvt/GaMS-9B-Instruct"
MODEL_NAME = "cjvt/GaMS-27B-Instruct"

model_to_dtype: dict[str, torch.dtype] = {
    "cjvt/GaMS-2B": torch.float32,
    "cjvt/GaMS-9B-Instruct": torch.bfloat16,
    "cjvt/GaMS-27B-Instruct": torch.bfloat16,
} 
PEFT_DIR = "/d/hpc/projects/onj_fri/nmlp/PEFT/"
# IO_PAIRS_PATH = "/d/hpc/projects/onj_fri/nmlp/dp2.jsonl"
# MAX_LENGTH: int = 512
IO_PAIRS_PATH = "/d/hpc/projects/onj_fri/nmlp/dp1.jsonl"
MAX_LENGTH: int = 1024

if not os.path.exists(PEFT_DIR):
    os.makedirs(PEFT_DIR, exist_ok=True)

RESUME_FROM = os.path.exists(os.path.join(PEFT_DIR, "adapter_model.bin"))
USE_CUDA = torch.cuda.is_available()

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ---------- LoRA Configuration ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none"
    )
    
    # ---------- Load Model ----------
    if RESUME_FROM:
        print("Resuming from PEFT checkpoint...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=model_to_dtype[MODEL_NAME],
        )
        model = PeftModel.from_pretrained(base_model, PEFT_DIR)
    else:
        print("Loading base model and applying LoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=model_to_dtype[MODEL_NAME],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    # Make sure the model is on device
    # model.to(device)
    # model.gradient_checkpointing_enable()
    
    # ---------- Actual Dataset ----------
    def get_dataset():
        data = {
            "input": [],
            "output": [],
        }
        with open(IO_PAIRS_PATH, "rt") as f:
            i = 0
            for line in f:
                data_point = json.loads(line)
                data["input"].append(data_point["input"])
                data["output"].append(data_point["output"])
                i += 1
        print(f"Loaded {i} input-output pairs from {IO_PAIRS_PATH}")
        return Dataset.from_dict(data)
    
    # ---------- Dummy Dataset ----------
    data = {
        "input": [
            "Kaj se dogaja na štajerski avtocesti danes?",
            "Promet na obvoznici v Ljubljani?",
        ],
        "output": [
            "Na štajerski avtocesti je zgoščen promet proti Mariboru.",
            "Na ljubljanski obvoznici so zastoji zaradi nesreče pri razcepu Kozarje.",
        ]
    }
    # dataset = Dataset.from_dict(data)
    dataset = get_dataset()
    
    # ---------- Preprocessing ----------
    def preprocess(example):
        prompt = example["input"]
        output = example["output"]
        full_input = f"VHOD:\n{prompt}\n\nIZHOD:\n{output}\n<EOS>"
        tokenized = tokenizer(full_input, truncation=True, padding="max_length", max_length=256)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def tokenize_function(example):
    
        # Step 1: Define the prompt template
        prompt = f"VHOD:\n{example['input']}\n\nIZHOD:\n"
        answer = f"{example["output"]}<EOS>"
    
        # Step 2: Tokenize prompt and answer SEPARATELY
        tokenized_prompt = tokenizer(prompt, truncation=True, add_special_tokens=False)
        tokenized_answer = tokenizer(answer, truncation=True, add_special_tokens=False)
    
        # Step 3: Combine and create labels
        input_ids = tokenized_prompt["input_ids"] + tokenized_answer["input_ids"]
        attention_mask = [1] * len(input_ids)  # All tokens are active
        labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_answer["input_ids"]
    
        # Step 4: Truncate if exceeds max_length
        if len(input_ids) > MAX_LENGTH:
            print(f"Warning: truncating input ({len(input_ids)})")
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    print("Preprocessing dataset...")
    dataset = dataset.map(tokenize_function)
    
    # ---------- Training Args ----------
    training_args = TrainingArguments(
        output_dir=f"{PEFT_DIR}/outputs",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=50,
        save_total_limit=2,
        learning_rate=1e-4,
        report_to="none",
        bf16=(model_to_dtype[MODEL_NAME] == torch.bfloat16),
        label_names=["labels"],
    )
    
    # data_collator = DataCollatorWithPadding(tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100  # Ignore padding in loss
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ---------- TRAIN ----------
    print("Starting training...")
    trainer.train()
    
    # ---------- SAVE ----------
    print("Saving PEFT model...")
    model.save_pretrained(PEFT_DIR)
    tokenizer.save_pretrained(PEFT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
