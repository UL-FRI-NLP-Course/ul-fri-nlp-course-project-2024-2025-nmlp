import pandas as pd

INPUT_DATA_PATH: str = "data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"

def load_data() -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame()
    sheet_to_df: dict[str, pd.DataFrame] = pd.read_excel(INPUT_DATA_PATH, sheet_name=None)
    for sheet_name, sheet_df in sheet_to_df.items():
        sheet_df["sheet_name"] = sheet_name
        df = pd.concat([df, sheet_df])
    return df

def main():
    df: pd.DataFrame = load_data()

if __name__ == "__main__":
    main()
