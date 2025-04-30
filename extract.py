import pandas as pd
import os
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from difflib import SequenceMatcher


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
    df = pd.read_excel("filtered_traffic_2022_01_31.xlsx")

    # Define which columns to clean (only Content* columns)
    columns_to_clean = [col for col in df.columns if col.startswith("Content")]

    def strip_html(text):
        if pd.isna(text):
            return ""
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ", strip=True)

    # Clean each target column
    for col in columns_to_clean:
        df[col] = df[col].apply(strip_html)

    # Save cleaned version
    df.to_excel("filtered_traffic_2022_01_31_cleaned.xlsx", index=False)
    print("Cleaned and saved to filtered_traffic_2022_01_30_cleaned.xlsx")

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
    with open("TMP_Jan31_outputs.txt", "w", encoding="utf-8") as f:
        for fname, content in parsed_outputs:
            f.write(f"--- {fname} ---\n{content}\n\n")

    print(f"\n✅ Done. Parsed {len(parsed_outputs)} RTF files.")


def match():

    # Load the cleaned Excel data
    excel_path = "/mnt/data/filtered_traffic_2022_01_30_cleaned.xlsx"
    df = pd.read_excel(excel_path)

    # Load the TMP output text
    with open("/mnt/data/TMP_Jan30_outputs.txt", encoding="utf-8") as f:
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


# extract_excel_by_date()
# extract_rtf()
# clean_excel()
match()