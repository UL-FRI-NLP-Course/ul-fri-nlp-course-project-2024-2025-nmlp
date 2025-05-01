import pandas as pd
import os
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import re
from sentenceMatching import *
import glob
from datetime import timedelta

def extract_excel_by_date(path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx", destination="filtered_traffic_2022_01_30.xlsx", 
                        from_date="2022-01-30 00:00:00", to_date="2022-01-30 23:59:59"):
    # 1. Load the Excel file
    df = pd.read_excel(path)

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
    print(f"Saved {len(filtered_df)} rows to 'filtered_traffic_2022_01_30.csv'")


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

def find_closest_rtf_and_extract(base_dir, target_datetime):
    # Build the folder path (e.g., Promet 2022/Januar 2022)
    month_translation = {
        "January": "Januar", "February": "Februar", "March": "Marec", "April": "April",
        "May": "Maj", "June": "Junij", "July": "Julij", "August": "Avgust",
        "September": "September", "October": "Oktober", "November": "November", "December": "December"
    }
    month_name = month_translation[target_datetime.strftime("%B")]
    year = target_datetime.strftime("%Y")
    folder = f"{base_dir}/Promet {year}/{month_name} {year}"
    print(folder)
    
    rtf_files = sorted(glob.glob(os.path.join(folder, "TMP*.rtf")), key=rtf_datetime_sort_key)

    for fname in rtf_files:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                content = rtf_to_text(f.read()).strip()
                return fname, content
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    return None, ""

def prepare_prompt_from_datetime(
    timestamp_str="2022-01-30 00:00:00",
    hours_back=3,
    rtf_base="C:/Users/a/Desktop/git/magisterij/1.2/NLP/RTVSlo/Podatki - rtvslo.si",
    excel_path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx",
    temp_excel="filtered_traffic.xlsx",
    temp_cleaned="filtered_traffic_cleaned.xlsx",
    prompt_output="prompt_input.xlsx",
    grouping_fn=group_unique_semantic_informative
):
    # Step 1: Parse target datetime
    target_time = pd.to_datetime(timestamp_str)

    # Step 2: Find RTF report closest to this time
    rtf_file, rtf_text = find_closest_rtf_and_extract(rtf_base, target_time)
    if not rtf_file:
        print("No RTF file found.")
        return

    print(f"Closest RTF: {rtf_file}")

    # Step 3: Extract Excel rows X hours before the RTF timestamp
    from_time = target_time - timedelta(hours=hours_back)
    extract_excel_by_date(path=excel_path, destination=temp_excel,
                          from_date=from_time.strftime('%Y-%m-%d %H:%M:%S'),
                          to_date=target_time.strftime('%Y-%m-%d %H:%M:%S'))

    # Step 4: Clean and group the Excel
    clean_excel(input=temp_excel, output=temp_cleaned)
    create_prompt_input(input=temp_cleaned, output=prompt_output, function=grouping_fn)

    print(f"\nFinal prompt-input saved to {prompt_output}")
    print(f"RTF output (should match that time):\n\n---\n{rtf_text}\n---\n")



prepare_prompt_from_datetime(
    timestamp_str="2022-01-30 15:00:00",
    hours_back=8
)


# extract_excel_by_date()
# extract_rtf()
# clean_excel()
# create_prompt_input(output='group_semantic.xlsx', function=group_unique_semantic)
# create_prompt_input(output='group_informative.xlsx', function=group_unique_semantic_informative)