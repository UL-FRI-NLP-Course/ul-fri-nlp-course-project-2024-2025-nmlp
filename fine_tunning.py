from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import load_dataset, Dataset
import torch
import os


# dummy preprocess function
def preprocess(example):
    model_input = tokenizer(example["input"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(example["output"], padding="max_length", truncation=True, max_length=256)
    model_input["labels"] = labels["input_ids"]
    return model_input

# Load base model
# ---------- CONFIGURATION ----------
MODEL_NAME = "cjvt/GaMS-2B"
PEFT_DIR = "./peft-gams2b"
RESUME_FROM = os.path.exists(os.path.join(PEFT_DIR, "adapter_model.bin"))
USE_CUDA = torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# configurations
# r:Rank of the LoRA matrices. Controls added parameter count. Typical values: 4, 8, 16
# lora_alpha: Scaling factor applied to LoRA weights. Typically: 16, 32, 64
# lora_dropout: Dropout rate applied to LoRA modules during training. Helps regularization.
# bias:     If 'none', only LoRA weights are trained; if 'all', biases too. Use 'none' unless you want more training overhead.

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_2_SEQ_LM,
    bias="none"
)


# PEFT configuration (LoRA)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # This matches "text2text-generation"
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# load model if exists
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if RESUME_FROM:
    print("Resuming from PEFT checkpoint...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, PEFT_DIR)
else:
    print("Loading base model and applying LoRA...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

# Prepare training data (dummy data for demonstration)
print("Preparing training data...")
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
dataset = Dataset.from_dict(data)


print("Preprocessing dataset...")
dataset = dataset.map(preprocess)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    evaluation_strategy="no",
    learning_rate=1e-4,
    report_to="none",
    fp16=USE_CUDA,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ---------- TRAINING ----------
print("Starting training...")
trainer.train()

# ---------- SAVE MODEL ----------
print("Saving PEFT model...")
model.save_pretrained(PEFT_DIR)
tokenizer.save_pretrained(PEFT_DIR)
print("Done!")
