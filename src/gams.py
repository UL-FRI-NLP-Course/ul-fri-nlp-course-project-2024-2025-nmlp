from transformers import pipeline
import torch

# model_id = "cjvt/GaMS-2B"
model_id = "cjvt/GaMS-27B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    # device_map="cuda",
    device_map="auto", # Multi-GPU
    model_kwargs={"torch_dtype": torch.bfloat16},
)

prompts = [
    "The examples of antonyms are:\nhigh => low\nwide => narrow\nbig =>",
    "Pristanek je bil prvi nadzorovani spust ameriškega vesoljskega plovila na površje Lune po Apollu 17 leta 1972, ko je na Luni pristala zadnja Nasina misija s posadko.\nDoslej so na Luni pristala vesoljska plovila le iz štirih drugih držav –",
    "U četvrtak je bila prva polufinalna večer Dore, a komentari na društvenim mrežama ne prestaju. U nedjeljno finale prošli su:"
]

sequences = pline(
    prompts,
    max_new_tokens=512,
    num_return_sequences=1
)

for seq in sequences:
    print("--------------------------")
    print(f"Result: {seq[0]['generated_text']}")
    print("--------------------------\n")

