import json

INPUTS_PATH: str = "data/dp1_inputs.jsonl"
OUTPUTS_PATH: str = "data/dp1_outputs.jsonl"
NEW_FILE: str = "data/dp1_gt.jsonl"

def main():
    with open(INPUTS_PATH, "rt") as file:
        inputs_lines: list[str] = file.readlines()
        inputs: list[dict] = [json.loads(line) for line in inputs_lines]
    with open(OUTPUTS_PATH, "rt") as file_src, open(NEW_FILE, "wt") as file_dst:
        for line in file_src.readlines():
            obj = json.loads(line)
            gt = None
            for obj_ in inputs:
                if obj_["text"] == obj["input"]:
                    gt = obj_["gt"]
            if gt:
                obj["gt"] = gt
                file_dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                print("Failed to find gt for line: " + line)

if __name__ == "__main__":
    main()

