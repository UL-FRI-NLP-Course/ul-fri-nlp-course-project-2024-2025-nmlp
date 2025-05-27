from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig

PEFT_DIR = "./peft-gams2b"  # path to the pretrained PEFT model 

# Load PEFT config to get base model
config = PeftConfig.from_pretrained(PEFT_DIR)
base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, PEFT_DIR)
tokenizer = AutoTokenizer.from_pretrained(PEFT_DIR)


from transformers import pipeline
pline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

prompt = "Kaj se dogaja na primorski avtocesti danes?"

result = pline(prompt, max_new_tokens=128, do_sample=False)
print("Prometno poročilo:", result[0]['generated_text'])


