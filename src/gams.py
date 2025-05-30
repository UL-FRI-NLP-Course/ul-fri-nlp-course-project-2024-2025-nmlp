from transformers import pipeline
import time
import json
import pandas as pd
from fine_tunning.dp2_inf import prompt_template
import torch

INPUTS_PATH: str = "data/dp2_inputs.jsonl"
OUTPUT_PATH: str = "data/basic_outputs.jsonl"

# model_id = "cjvt/GaMS-2B"
# model_id = "cjvt/GaMS-9B-Instruct"
model_id = "cjvt/GaMS-27B-Instruct"

model_to_dtype: dict[str, torch.dtype] = {
    "cjvt/GaMS-2B": torch.float32,
    "cjvt/GaMS-9B-Instruct": torch.bfloat16,
    "cjvt/GaMS-27B-Instruct": torch.bfloat16,
} 

pline = pipeline(
    "text-generation",
    model=model_id,
    # device_map="cuda",
    device_map="auto", # Multi-GPU
    torch_dtype=model_to_dtype[model_id],
)

def main():
    lines_to_skip = 0
    with open(OUTPUT_PATH, "rt") as file:
        lines_to_skip = len(file.readlines())
    print(f"Lines to skip: {lines_to_skip}")
    df: pd.DataFrame = pd.read_json(INPUTS_PATH, lines=True)
    example_inputs: list[str] = [x for x in df["text"]]
    example_inputs = example_inputs[lines_to_skip:]
    prompts: list[str] = [prompt_template.format(input=example) for example in example_inputs]
    with open(OUTPUT_PATH, "at") as file:
        for i in range(len(prompts)):
            print(f"[{i+1}/{len(prompts)}]")
            t0 = time.time()
            sequences = pline(
                [prompts[i]],
                max_new_tokens=512,
                num_return_sequences=1,
                return_full_text=False,
            )
            try:
                seq = sequences[0]
                result: str = seq[0]["generated_text"]
                result = result[:result.index("<EOS>")]
                file.write(json.dumps({"input": example_inputs[i], "output": result}, ensure_ascii=False) + "\n")
                file.flush()
            except Exception as e:
                print("Error while trying to save generated output: " + str(e))
            print(f"Took {time.time()-t0:.1f} seconds")
    print("Done")

if __name__ == "__main__":
    main()
