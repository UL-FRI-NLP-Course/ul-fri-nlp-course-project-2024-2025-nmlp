import pandas as pd
import os
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import re

def extract_excel_by_date(path="data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx", destination="filtered_traffic_2022_01_30.xlsx", 
                        from_date="2022-01-30 00:00:00", to_date="2022-01-30 23:59:59"):
    # 1. Load the Excel file
    df = pd.read_excel(path)

    # 2. Convert 'Datum' to datetime (accounting for double space)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y  %H:%M:%S', errors='coerce')

    # 3. Define the time window for Jan 31st, 2022
    start_time = pd.Timestamp(from_date)
    end_time = pd.Timestamp(to_date)

    # 4. Filter rows within that time range
    filtered_df = df[(df['Datum'] >= start_time) & (df['Datum'] <= end_time)]

    # 5. Save the filtered data to a new CSV
    filtered_df.to_excel(destination, index=False)

    # 6. Confirm
    print(f"Saved {len(filtered_df)} rows to 'filtered_traffic_2022_01_30.csv'")


def clean_excel():
    # Load Excel
    df = pd.read_excel("filtered_traffic_2022_01_30.xlsx")

    # Clean all string-like columns (not just Content*)
    text_columns = [col for col in df.columns if df[col].dtype == object]

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
    output_path = "filtered_traffic_2022_01_30_cleaned.xlsx"
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
                print(f"✓ Read {filename}")
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    # Optional: save all outputs to one file
    with open("TMP_Jan30_outputs.txt", "w", encoding="utf-8") as f:
        for fname, content in parsed_outputs:
            f.write(f"--- {fname} ---\n{content}\n\n")

    print(f"\n Done. Parsed {len(parsed_outputs)} RTF files.")


def match():
    # Load the cleaned Excel data
    excel_path = "filtered_traffic_2022_01_30_cleaned.xlsx"
    df = pd.read_excel(excel_path)

    # Load the TMP output text
    with open("TMP_Jan30_outputs.txt", encoding="utf-8") as f:
        tmp_text = f.read()

    # Relevant content columns to consider for matching
    content_columns = [col for col in df.columns if col.startswith("Content")]

    # Flatten all TMP texts into a list of lines for similarity matching
    tmp_lines = [line.strip() for line in tmp_text.splitlines() if line.strip() and not line.startswith("---")]

    # Define function to score similarity
    def max_similarity(content, candidates):
        return max(SequenceMatcher(None, content, candidate).ratio() for candidate in candidates)

    # Score each row based on maximum similarity of any of its content columns to TMP lines
    matches = []

    for idx, row in df.iterrows():
        for col in content_columns:
            content = str(row[col])
            if content.strip():
                score = max_similarity(content, tmp_lines)
                if score > 0.6:  # threshold for "strong" match
                    matches.append((score, idx, col, content))
                    break  # only take first good match per row

    # Sort by score descending and select top N unique rows
    matches = sorted(matches, reverse=True)
    selected_indices = sorted(set([m[1] for m in matches[:10]]))  # select top 10 unique rows

    # Extract those rows
    matched_df = df.loc[selected_indices]

    matched_df.to_excel("matched_traffic_2022_01_30.xlsx", index=False)


# Clean each text cell: strip HTML and normalize whitespace
def clean_text(text):
    if isinstance(text, str):
        # Strip HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        # Normalize spaces
        return re.sub(r'\s+', ' ', text).strip()
    return ""

def create_prompt_input():
    # Reload the cleaned Excel file
    df = pd.read_excel("filtered_traffic_2022_01_30_cleaned.xlsx")

    # Drop unused columns
    df = df.drop(columns=["LegacyId", "Operater"], errors="ignore")

    # Round timestamps to 30-minute intervals
    df["Datum"] = pd.to_datetime(df["Datum"])
    df["TimeGroup"] = df["Datum"].dt.floor("30min")


    # Apply cleaning across all columns (except time metadata)
    text_columns = df.columns.difference(["Datum", "TimeGroup"])
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    # Combine all cleaned fields into a deduplicated list of sentences per row
    def extract_unique_sentences(row):
        joined = " ".join(str(cell) for cell in row if isinstance(cell, str))
        sentences = re.split(r'(?<=[.!?])\s+', joined)
        unique = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
        return " ".join(unique)

    df["combined"] = df[text_columns].apply(extract_unique_sentences, axis=1)

    # Group by time block and remove cross-row duplicates too
    def group_unique_sentences(series):
        all_sentences = []
        for text in series:
            all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
        unique = list(dict.fromkeys([s.strip() for s in all_sentences if s.strip()]))
        return " ".join(unique)

    grouped = (
        df.groupby("TimeGroup")["combined"]
        .apply(group_unique_sentences)
        .reset_index()
        .rename(columns={"combined": "data"})
    )

    grouped.to_excel("prompt_input_traffic_2022_01_30.xlsx", index=False)
    print("Prompt input saved to 'prompt_input_traffic_2022_01_30.xlsx'")



# extract_excel_by_date()
# extract_rtf()
# clean_excel()
# match()
create_prompt_input()