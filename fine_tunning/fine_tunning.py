from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}", flush=True)

# ---------- CONFIGURATION ----------
# MODEL_NAME = "cjvt/GaMS-2B"
MODEL_NAME = "cjvt/GaMS-27B-Instruct"
PEFT_DIR = "/d/hpc/projects/onj_fri/nmlp/PEFT/"
IO_PAIRS_PATH = "/d/hpc/projects/onj_fri/nmlp/dp2.jsonl"

if not os.path.exists(PEFT_DIR):
    os.makedirs(PEFT_DIR, exist_ok=True)

RESUME_FROM = os.path.exists(os.path.join(PEFT_DIR, "adapter_model.bin"))
USE_CUDA = torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------- LoRA Configuration ----------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# ---------- Load Model ----------
if RESUME_FROM:
    print("Resuming from PEFT checkpoint...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, PEFT_DIR)
else:
    print("Loading base model and applying LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

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
            data["input"].append(data_point["vhod"])
            data["output"].append(data_point["izhod"])
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
    full_input = f"{prompt}\n###\n{output}"  # join prompt/output for causal LM
    tokenized = tokenizer(full_input, truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Preprocessing dataset...")
dataset = dataset.map(preprocess)

# ---------- Training Args ----------
training_args = TrainingArguments(
    output_dir=f"{PEFT_DIR}/outputs",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    learning_rate=1e-4,
    report_to="none",
    fp16=USE_CUDA,
)

data_collator = DataCollatorWithPadding(tokenizer)

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
