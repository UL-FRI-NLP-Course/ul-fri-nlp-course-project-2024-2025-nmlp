import pandas as pd
import os
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import re
from sentenceMatching import *
import glob
from datetime import timedelta
import time
import json
from multiprocessing import Pool, cpu_count
from datetime import datetime
import random

# Filter Excel by datetime range
def extract_excel_by_date(path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx", destination="filtered_traffic_2022_01_30.xlsx", 
                        from_date="2022-01-30 00:00:00", to_date="2022-01-30 23:59:59"):
    # 1. Load the Excel file
    all_sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(all_sheets.values(), ignore_index=True)

    # 2. Convert 'Datum' to datetime (accounting for double space)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y  %H:%M:%S', errors='coerce')

    # remove TitleDeloNaCestiSLO
    df = df.drop(columns=["TitleDeloNaCestiSLO"], errors='ignore')

    # 3. Define the time window for Jan 31st, 2022
    start_time = pd.Timestamp(from_date)
    end_time = pd.Timestamp(to_date)

    # 4. Filter rows within that time range
    filtered_df = df[(df['Datum'] >= start_time) & (df['Datum'] <= end_time)]

    # 5. Save the filtered data to a new CSV
    filtered_df.to_excel(destination, index=False)

    # 6. Confirm
    print(f"Saved {len(filtered_df)} rows to {path}")

# Clean and strip HTML from Excel content
def clean_excel(input='filtered_traffic_2022_01_30.xlsx', output="filtered_traffic_2022_01_30_cleaned.xlsx"):
    # Load Excel
    df = pd.read_excel(input)

    # Clean all string-like columns (not just Content*)
    text_columns = [col for col in df.columns if df[col].dtype == object]

    # remove if exists TitleDeloNaCestiSLO
    if "TitleDeloNaCestiSLO" in text_columns:
        text_columns.remove("TitleDeloNaCestiSLO")

    def strip_html(text):
        if pd.isna(text):
            return ""
        # Fully strip HTML tags
        soup = BeautifulSoup(str(text), "html.parser")
        text_only = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text_only = re.sub(r'\s+', ' ', text_only)
        return text_only.strip()

    # Apply cleaning
    for col in text_columns:
        df[col] = df[col].apply(strip_html)

    # Save cleaned version
    output_path = output
    df.to_excel(output_path, index=False)
    print(f"Cleaned and saved to {output_path}")

# Parse .rtf files from RTVSlo reports
def extract_rtf():
    # Define path to RTF folder
    base_path = r"C:/Users/a/Desktop/git/magisterij/1.2/NLP/RTVSlo/Podatki - rtvslo.si/Promet 2022/Januar 2022"

    # RTF range: TMP-26.rtf to TMP-48.rtf
    start_index = 26
    end_index = 48

    # Collect parsed outputs
    parsed_outputs = []

    for i in range(start_index, end_index + 1):
        filename = f"TMP-{i}.rtf"
        full_path = os.path.join(base_path, filename)

        try:
            with open(full_path, "r", encoding="utf-8") as file:
                rtf_content = file.read()
                plain_text = rtf_to_text(rtf_content)
                parsed_outputs.append((filename, plain_text.strip()))
                print(f"Read {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Optional: save all outputs to one file
    with open("TMP_Jan30_outputs.txt", "w", encoding="utf-8") as f:
        for fname, content in parsed_outputs:
            f.write(f"--- {fname} ---\n{content}\n\n")

    print(f"\n Done. Parsed {len(parsed_outputs)} RTF files.")


# Clean each text cell: strip HTML and normalize whitespace
def clean_text(text):
    if isinstance(text, str):
        # Strip HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        # Normalize spaces
        return re.sub(r'\s+', ' ', text).strip()
    return ""

# Combine all cleaned fields into a deduplicated list of sentences per row
def extract_unique_sentences(row):
    joined = " ".join(str(cell) for cell in row if isinstance(cell, str))
    sentences = re.split(r'(?<=[.!?])\s+', joined)
    unique = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
    return " ".join(unique)

# Group by time block and remove cross-row duplicates too
def group_unique_sentences(series):
    all_sentences = []
    for text in series:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    unique = list(dict.fromkeys([s.strip() for s in all_sentences if s.strip()]))
    return " ".join(unique)

# Group and clean Excel data for prompt generation
def create_prompt_input(input='filtered_traffic_2022_01_30_cleaned.xlsx', output='prompt_input_traffic_2022_01_30.xlsx', function=group_unique_sentences):
    # Reload the cleaned Excel file
    df = pd.read_excel(input)

    # Drop unused columns
    df = df.drop(columns=["LegacyId", "Operater"], errors="ignore")

    # Round timestamps to 30-minute intervals
    df["Datum"] = pd.to_datetime(df["Datum"])
    df["TimeGroup"] = df["Datum"].dt.floor("30min")

    # Apply cleaning across all columns (except time metadata)
    text_columns = df.columns.difference(["Datum", "TimeGroup"])
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    df["combined"] = df[text_columns].apply(extract_unique_sentences, axis=1)

    grouped = (
        df.groupby("TimeGroup")["combined"]
        .apply(function)
        .reset_index()
        .rename(columns={"combined": "data"})
    )

    grouped.to_excel(output, index=False)
    print("Prompt input saved to", output)


import glob
from datetime import timedelta

def rtf_datetime_sort_key(filename):
    # Extract numeric suffix from TMP-XXX.rtf or TMP.rtf
    match = re.search(r'TMP-(\d+)', filename)
    if match:
        return -int(match.group(1))  # reverse order
    elif 'TMP.rtf' in filename:
        return float('inf')  # oldest
    else:
        return float('-inf')

def parse_rtf_datetime(text):
    # Look for datetime in RTF like: "01. 01. 2022 08.30"
    match = re.search(r'(\d{2}\. ?\d{2}\. ?\d{4})\s+(\d{2}\.\d{2})', text)
    if match:
        date_str, time_str = match.groups()
        dt_str = date_str.replace(" ", "") + " " + time_str.replace(".", ":")
        try:
            return pd.to_datetime(dt_str, format="%d.%m.%Y %H:%M")
        except:
            return None
    return None


def find_closest_rtf_and_extract(base_dir, target_datetime):
    month_translation = {
        "January": "Januar", "February": "Februar", "March": "Marec", "April": "April",
        "May": "Maj", "June": "Junij", "July": "Julij", "August": "Avgust",
        "September": "September", "October": "Oktober", "November": "November", "December": "December"
    }
    month_name = month_translation[target_datetime.strftime("%B")]
    year = target_datetime.strftime("%Y")
    folder = f"{base_dir}/Promet {year}/{month_name} {year}"
    print(f"Searching in folder: {folder}")
    
    rtf_files = glob.glob(os.path.join(folder, "TMP*.rtf"))

    best_match = None
    smallest_diff = timedelta.max

    for fname in rtf_files:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                text = rtf_to_text(f.read())
                dt = parse_rtf_datetime(text)
                if dt is None:
                    continue
                diff = abs(target_datetime - dt)
                if diff < smallest_diff:
                    best_match = (fname, text.strip())
                    smallest_diff = diff
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    if best_match:
        print(f"Closest RTF: {best_match[0]} at {smallest_diff}")
        return best_match
    return None, ""

def create_flat_input(input='filtered_traffic_cleaned.xlsx', output='flat_prompt_input.txt', function=group_unique_semantic_informative, from_time=None):
    df = pd.read_excel(input)

    # Clean & flatten
    df["Datum"] = pd.to_datetime(df["Datum"])
    text_columns = df.columns.difference(["Datum"])
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    combined_series = df[text_columns].apply(extract_unique_sentences, axis=1)
    flat_body = function(combined_series)

    # Format header based on from_time
    if from_time is None:
        from_time = df["Datum"].min()
    header = f"Prometne informacije       {from_time.strftime('%d. %m. %Y')}   \t   {from_time.strftime('%H.%M')}           2. program"

    full_text = f"{header}\n\nPodatki o prometu.\n\n{flat_body}"

    with open(output, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved flat input to {output}")
    return full_text

def prepare_prompt_from_datetime(
    timestamp_str="2022-01-30 00:00:00",
    hours_back=3,
    rtf_base="/Podatki - rtvslo.si",
    excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    temp_excel="filtered_traffic.xlsx",
    temp_cleaned="filtered_traffic_cleaned.xlsx",
    prompt_output="prompt_input.xlsx",
    flat_output="flat_prompt_input.txt",
    grouping_fn=group_unique_semantic_informative
):
    # Step 1: Parse target datetime
    target_time = pd.to_datetime(timestamp_str)

    # Step 2: Find RTF report closest to this time
    rtf_file, rtf_text = find_closest_rtf_and_extract(rtf_base, target_time)
    if not rtf_file:
        print("No RTF file found.")
        return "",""

    print(f"Closest RTF: {rtf_file}")

    # Step 3: Extract Excel rows X hours before the RTF timestamp
    from_time = target_time - timedelta(hours=hours_back)
    extract_excel_by_date(path=excel_path, destination=temp_excel,
                          from_date=from_time.strftime('%Y-%m-%d %H:%M:%S'),
                          to_date=target_time.strftime('%Y-%m-%d %H:%M:%S'))

    # Step 4: Clean and group the Excel
    clean_excel(input=temp_excel, output=temp_cleaned)
    # create_prompt_input(input=temp_cleaned, output=prompt_output, function=grouping_fn)
    flat_input = create_flat_input(input=temp_cleaned, output=flat_output, function=grouping_fn, from_time=target_time)

    print(f"\nFinal prompt-input saved to {prompt_output}")
    print(f"RTF output (should match that time):\n\n---\n{rtf_text}\n---\n")
    return flat_input, rtf_text


def get_prompt_and_output(timestamp_str, hours_back, rtf_base, grouping_fn=group_unique_semantic_informative):
    prompt_input, rtf_output = prepare_prompt_from_datetime(
        timestamp_str=timestamp_str,
        hours_back=hours_back,
        rtf_base=rtf_base,
        grouping_fn=grouping_fn
    )
    return prompt_input, rtf_output

def prepare_input_output_pairs(start_date="2022-01-2 12:30:00", end_date="2024-12-2 23:30:00", hours_back=24, rtf_base="/RTVSlo/Podatki - rtvslo.si", json_path="input_output_pairs.json"):
    current_date = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    pairs = []

    # Make sure hours_back is iterable
    hours_back_iter = [hours_back] if isinstance(hours_back, int) else hours_back

    while current_date <= end_date_dt:
        print(f"Processing date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        for hours in hours_back_iter:
            prompt_input, rtf_output = get_prompt_and_output(
                timestamp_str=current_date.strftime('%Y-%m-%d %H:%M:%S'),
                hours_back=hours,
                rtf_base=rtf_base
            )
            if prompt_input and rtf_output:
                pairs.append({
                    "input": prompt_input,
                    "output": rtf_output
                })
        current_date += timedelta(hours=1)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(pairs)} input-output pairs to '{json_path}'")
    return pairs


def process_rtf_file(args):
    rtf_path, start_dt, end_dt, df, hours_back = args
    try:
        with open(rtf_path, 'r', encoding='utf-8') as f:
            rtf_text = rtf_to_text(f.read())
            rtf_time = parse_rtf_datetime(rtf_text)
            if rtf_time and (start_dt <= rtf_time <= end_dt):
                from_time = rtf_time - timedelta(hours=hours_back)
                to_time = rtf_time
                subset = df[(df['Datum'] >= from_time) & (df['Datum'] <= to_time)]
                if subset.empty:
                    return None
                flat_input = group_unique_semantic_informative(subset['combined'])
                if not flat_input.strip():
                    return None
                prompt = f"Prometne informacije {to_time.strftime('%d. %m. %Y')} {to_time.strftime('%H.%M')}\n\n{flat_input}"
                return {"input": prompt, "output": rtf_text}
    except Exception as e:
        print(f"Failed to parse {rtf_path}: {e}")
    return None

def generate_input_output_pairs_fast(
    excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    rtf_base="RTVSlo/Podatki - rtvslo.si",
    start_date="2022-01-30 00:00:00",
    end_date="2022-01-30 23:59:59",
    hours_back=3,
    json_path="input_output_pairs.json",
    grouping_fn=group_unique_semantic_informative
):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Load Excel and preprocess
    df = pd.read_excel(excel_path)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y  %H:%M:%S', errors='coerce')
    df = df.drop(columns=["TitleDeloNaCestiSLO", "LegacyId", "Operater"], errors='ignore')

    text_columns = df.columns.difference(['Datum'])
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    df['combined'] = df[text_columns].apply(extract_unique_sentences, axis=1)

    # Parse RTFs within date range
    rtf_files = glob.glob(os.path.join(rtf_base, "**", "TMP*.rtf"), recursive=True)
    parsed_rtf = []
    for rtf_path in sorted(rtf_files, key=rtf_datetime_sort_key):
        try:
            with open(rtf_path, 'r', encoding='utf-8') as f:
                rtf_text = rtf_to_text(f.read())
                rtf_time = parse_rtf_datetime(rtf_text)
                if rtf_time and (start_dt <= rtf_time <= end_dt):
                    parsed_rtf.append((rtf_time, rtf_text.strip()))
        except Exception as e:
            print(f"Failed to parse {rtf_path}: {e}")

    # Create input-output pairs
    all_pairs = []
    for rtf_time, rtf_text in parsed_rtf:
        from_time = rtf_time - timedelta(hours=hours_back)
        to_time = rtf_time

        # Sliding window
        subset = df[(df['Datum'] >= from_time) & (df['Datum'] <= to_time)]
        if subset.empty:
            continue

        flat_input = grouping_fn(subset['combined'])
        if not flat_input.strip():
            continue

        prompt = f"Prometne informacije {to_time.strftime('%d. %m. %Y')} {to_time.strftime('%H.%M')}\n\n{flat_input}"
        all_pairs.append({
            "input": prompt,
            "output": rtf_text
        })

    # Save to JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_pairs)} pairs to '{json_path}'")
    print(all_pairs)
    return all_pairs


# Test preprocessing strategies to see which has best similarity with RTF outputs
def test_preprocessing_strategies():
    strategies = [
        group_unique_sentences,
        group_unique_semantic_informative,
        group_tf_idf_informative,
        group_with_named_entity_preference
    ]
    for fn in strategies:
        for hours in [1, 2, 3, 4, 6, 8]:
            print(f"== Testing {fn.__name__} with {hours}h window ==")
            prompt_input, rtf_output = prepare_prompt_from_datetime(
                timestamp_str="2022-01-30 15:30:00",
                hours_back=hours,
                grouping_fn=fn
            )
            score = compute_similarity_score(prompt_input, rtf_output)
            print(f"{fn.__name__} @ {hours}h => Similarity: {score:.4f}\n")


def main_random_pairs(
    n_times=1,
    start_str="2023-01-01 00:00:00",
    end_str="2024-12-31 23:59:59",
    hours_back=4,
    rtf_base="/RTVSlo/Podatki - rtvslo.si",
    excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    jsonl_path="dp1.jsonl",
    grouping_fn=group_unique_semantic_informative
):
    """
    Generate `n_times` random timestamps between `start_str` and `end_str`,
    create input-output pairs via `prepare_prompt_from_datetime`, and
    write them as JSON lines to `jsonl_path`.
    """
    # Parse range
    start_dt = datetime.fromisoformat(start_str)
    end_dt = datetime.fromisoformat(end_str)
    delta = end_dt - start_dt

    pairs = []
    for i in range(n_times):
        # Generate random datetime in range
        rand_sec = random.random() * delta.total_seconds()
        ts = start_dt + timedelta(seconds=rand_sec)
        timetime = ts.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{i+1}/{n_times}] Processing timestamp: {timetime}")

        # Prepare prompt and output
        flat_input, rtf_output = prepare_prompt_from_datetime(
            timestamp_str=timetime,
            hours_back=hours_back,
            rtf_base=rtf_base,
            excel_path=excel_path,
            grouping_fn=grouping_fn,
            temp_excel=f"temp_{i}.xlsx",
            temp_cleaned=f"temp_cleaned_{i}.xlsx",
            prompt_output=f"prompt_{i}.xlsx",
            flat_output=f"flat_prompt_{i}.txt",
        )
        if flat_input and rtf_output:
            pairs.append({"input": flat_input, "output": rtf_output})
        else:
            print(f"    Skipped timestamp {timetime}: no data generated.")

    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} pairs to {jsonl_path}")
    return pairs

if __name__ == "__main__":
    start_time = time.time()
    # Generate random pairs used for inference and evaluation
    main_random_pairs()
    elapsed = time.time() - start_time
    print(f"Total time: {timedelta(seconds=elapsed)}")

    # Example usages:

    # extract_excel_by_date:
    # Filters Excel rows by a datetime window and saves to a new file.
    # extract_excel_by_date(
    #     path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    #     destination="filtered_jan30.xlsx",
    #     from_date="2022-01-30 00:00:00",
    #     to_date="2022-01-30 23:59:59"
    # )

    # clean_excel:
    # Strips HTML and normalizes whitespace in text columns.
    # clean_excel(
    #     input="filtered_jan30.xlsx",
    #     output="filtered_jan30_cleaned.xlsx"
    # )

    # extract_rtf:
    # Reads and extracts plain text from multiple RTF files.
    # extract_rtf()

    # create_prompt_input:
    # Groups and cleans text from Excel into 30-minute time blocks.
    # create_prompt_input(
    #     input="filtered_jan30_cleaned.xlsx",
    #     output="prompt_grouped.xlsx",
    #     function=group_unique_sentences
    # )

    # find_closest_rtf_and_extract:
    # Finds the RTF file closest to a given datetime.
    # find_closest_rtf_and_extract(
    #     base_dir="C:/Users/a/Desktop/git/magisterij/1.2/NLP/RTVSlo/Podatki - rtvslo.si",
    #     target_datetime=pd.Timestamp("2022-01-30 15:30:00")
    # )

    # create_flat_input:
    # Generates a flat prompt string from cleaned Excel data.
    # create_flat_input(
    #     input="filtered_jan30_cleaned.xlsx",
    #     output="flat_prompt.txt",
    #     function=group_unique_semantic_informative,
    #     from_time=pd.Timestamp("2022-01-30 15:30:00")
    # )

    # prepare_prompt_from_datetime:
    # Full pipeline: aligns RTF with Excel and generates prompt-output pair.
    # prepare_prompt_from_datetime(
    #     timestamp_str="2022-01-30 15:30:00",
    #     hours_back=4,
    #     excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"
    # )

    # get_prompt_and_output:
    # Wrapper to just get strings (prompt + RTF) for a given time.
    # get_prompt_and_output(
    #     timestamp_str="2022-01-30 15:30:00",
    #     hours_back=4,
    # )

    # prepare_input_output_pairs:
    # Loops through time window and saves prompt-output pairs to JSON.
    # prepare_input_output_pairs(
    #     start_date="2022-01-30 00:00:00",
    #     end_date="2022-01-30 23:59:59",
    #     hours_back=4,
    #     json_path="pairs.json"
    # )

    # generate_input_output_pairs_fast:
    # Faster batch method for creating all pairs between Excel and RTFs.
    # generate_input_output_pairs_fast(
    #     excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    #     start_date="2022-01-30 00:00:00",
    #     end_date="2022-01-30 23:59:59",
    #     hours_back=4,
    #     json_path="fast_pairs.json"
    # )

    # test_preprocessing_strategies:
    # Evaluates different grouping strategies on similarity to RTF.
    # test_preprocessing_strategies()

    # main_random_pairs:
    # Randomly samples N timestamps and creates JSONL of prompt-output pairs.
    # main_random_pairs(
    #     n_times=10,
    #     start_str="2023-01-01 00:00:00",
    #     end_str="2024-12-31 23:59:59",
    #     hours_back=3,
    #     excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    #     jsonl_path="random_output.jsonl"
    # )



