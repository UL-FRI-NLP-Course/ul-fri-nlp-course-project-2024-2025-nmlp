import pandas as pd
from src.input_data import load_data, InputReport

SEED: int = 42
NUM_SAMPLES: int = 100
OUTPUT_PATH: str = "data/dp1_inputs.jsonl"

def preprocess(row: pd.Series) -> str:
    report: InputReport = InputReport(row)
    s: str = "\n".join(par.raw for par in report.paragraphs)
    return s

def main():
    df: pd.DataFrame = load_data()
    subset: pd.DataFrame = df.sample(n=NUM_SAMPLES, random_state=SEED)
    subset["text"] = subset.apply(preprocess, axis=1)
    subset = pd.DataFrame(subset["text"])
    subset.to_json(OUTPUT_PATH, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()
