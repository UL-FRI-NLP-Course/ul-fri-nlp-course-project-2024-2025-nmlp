# Use a pipeline as a high-level helper
from transformers import pipeline

pline = pipeline("text2text-generation", model="cjvt/GaMS-2B")
# test
print(pline("Kaj se dogaja na štajerski avtocesti danes?"))

# pline = pipeline("text-generation", model="cjvt/GaMS-2B")

print("Model loaded!\n")
print(pline)

print("Loading prompt!\n")
# Read prompt from file
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read().strip()

print(prompt)
print("Prompt loaded!\n")
# Generate output
print("Generating report...\n")
output = pline(
    prompt,
    max_new_tokens=512,
    num_return_sequences=1
)


# Show result
print("Prometno poročilo:")
for out in output:
    print(output[0]['generated_text'])

