import re
import glob
import datetime
import itertools
import pandas as pd
from typing import Any
from striprtf.striprtf import rtf_to_text

PATH_TO_RTFS: str = "RTVSlo/Podatki - rtvslo.si"
OUTPUT_FILE: str = "data/rtfs_merged.jsonl"
HEADER_REGEX: re.Pattern = re.compile(r"^([^0-9]*?)\s*([0-9]+)\.\s*([0-9]+)\.\s*([0-9]+)\s*([0-9]+)\.([0-9]+)\s*(.*?)$")

def clean_str(to_be_cleaned: str) -> str:
    return re.sub(r"\x00", "", to_be_cleaned).strip()

def main():
    files: list[str] = glob.glob(f"{PATH_TO_RTFS}/**/*.rtf", recursive=True)
    rows: list[dict[str, Any]] = []
    for filename in files:
        with open(filename, "rt") as file:
            text: str = rtf_to_text(file.read()).strip()
            if len(text) < 10:
                print(f"File too short: {len(text)=}, {filename=}")
                continue
            header, body = text.split("\n", maxsplit=1)
            header, body = clean_str(header), clean_str(body)
            if len(list(itertools.islice(re.finditer(r"\s+", body), 5))) != 5:
                print(f"Body has too few sentences, skipping {filename=}")
                continue
            header_match: re.Match[str] | None = re.search(HEADER_REGEX, header)
            if header_match is None:
                print(f"Error while parsing {filename=}")
                continue
            prefix, day, month, year, hour, minute, postfix = header_match.groups()
            rows.append({
                "timestamp": datetime.datetime(year=int(year) if len(year) > 2 else (2000+int(year)), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=0),
                "header_prefix": prefix,
                "header_postfix": postfix,
                "body": body,
                "filename": re.sub(PATH_TO_RTFS, "", filename),
            })

    df: pd.DataFrame = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_json(OUTPUT_FILE, lines=True, orient="records", force_ascii=False)

if __name__ == "__main__":
    main()
    # df: pd.DataFrame = pd.read_json(OUTPUT_FILE, lines=True)

