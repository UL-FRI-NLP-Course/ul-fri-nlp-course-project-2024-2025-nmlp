import json

INPUTS_PATH: str = "data/dp1_inputs.jsonl"
OUTPUTS_PATH: str = "data/dp1_outputs.jsonl"
NEW_FILE: str = "data/dp1_gt.jsonl"

def main():
    # read inputs as UTF-8
    with open(INPUTS_PATH, "rt", encoding="utf-8") as file:
        inputs = [json.loads(line) for line in file]
        
    # read outputs as UTF-8, write new file as UTF-8
    with open(OUTPUTS_PATH, "rt", encoding="utf-8") as file_src, \
         open(NEW_FILE,     "wt", encoding="utf-8") as file_dst:

        for line in file_src:
            obj = json.loads(line)
            # find matching gt
            for inp in inputs:
                if inp["text"] == obj["input"]:
                    obj["gt"] = inp["gt"]
                    break

            if "gt" in obj:
                file_dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                print("Failed to find gt for line:", obj["input"])


if __name__ == "__main__":
    main()

