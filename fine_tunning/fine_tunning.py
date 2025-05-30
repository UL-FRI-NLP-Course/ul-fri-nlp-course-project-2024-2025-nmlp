from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset
import torch
import json
import os
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "auto" if torch.cuda.device_count() > 1 else device
print(f"Running on: {device}", flush=True)

# avoid duplication
# Trainer._setup_devices = lambda self: None

# ---------- CONFIGURATION ----------
# MODEL_NAME = "cjvt/GaMS-2B"
MODEL_NAME = "cjvt/GaMS-9B-Instruct"
# MODEL_NAME = "cjvt/GaMS-27B-Instruct"

model_to_dtype: dict[str, torch.dtype] = {
    "cjvt/GaMS-2B": torch.float32,
    "cjvt/GaMS-9B-Instruct": torch.bfloat16,
    "cjvt/GaMS-27B-Instruct": torch.bfloat16,
} 
# PEFT_DIR = "/d/hpc/projects/onj_fri/nmlp/PEFT/"
PEFT_DIR = "/d/hpc/projects/onj_fri/peft_ah/9B-instr-dp11"
IO_PAIRS_PATH = "/d/hpc/projects/onj_fri/nmlp/dp1.jsonl"

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
            attn_implementation="eager",  # For Gemma2 recommended
            device_map=device,
            torch_dtype=model_to_dtype[MODEL_NAME],
        )
        model = PeftModel.from_pretrained(base_model, PEFT_DIR)
    else:
        print("Loading base model and applying LoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation="eager",  # For Gemma2 recommended
            device_map=device,
            torch_dtype=model_to_dtype[MODEL_NAME],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    # Make sure the model is on device
    model.to(device)
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
    IGNORE_INDEX = -100
    MAX_LENGTH = 1024

    def preprocess(example):
        prompt = example["input"].strip()
        output = example["output"].strip()

        # Structured full prompt
        full_input = f"VHOD:\n{prompt}\n\nIZHOD:\n{output}\n<EOS>"
        full_prompt = f"VHOD:\n{prompt}\n\nIZHOD:\n"

        tokenized = tokenizer(
            full_input,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        # Mask prompt part in labels (we only want model to learn to predict the output)
        prompt_ids = tokenizer(
            full_prompt,
            truncation=True,
            max_length=MAX_LENGTH
        )["input_ids"]

        labels = tokenized["input_ids"].copy()
        masked_len = min(len(prompt_ids), len(labels))
        labels[:masked_len] = [IGNORE_INDEX] * masked_len

        tokenized["labels"] = labels
        return tokenized
    
    print("Preprocessing dataset...")
    dataset = dataset.map(preprocess, num_proc=4)
    
    # ---------- Training Args ----------
    training_args = TrainingArguments(
        output_dir=f"{PEFT_DIR}/outputs",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=20,
        learning_rate=1e-4,
        report_to="none",
        fp16=(model_to_dtype[MODEL_NAME] == torch.float16),
        bf16=(model_to_dtype[MODEL_NAME] == torch.bfloat16),
        disable_tqdm=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=4,
    )
    
    # data_collator = DataCollatorWithPadding(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
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
