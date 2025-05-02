import pandas as pd
import datetime
import re

INPUT_DATA_PATH: str = "data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"

def row2str(row: pd.Series) -> str:
    old_value: int | None = pd.get_option("display.max_colwidth")
    pd.set_option("display.max_colwidth", None)
    text: str = re.sub(" +", " ", str(row))
    pd.set_option("display.max_colwidth", old_value)
    return text

def load_data() -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame()
    sheet_to_df: dict[str, pd.DataFrame] = pd.read_excel(INPUT_DATA_PATH, sheet_name=None)
    for sheet_name, sheet_df in sheet_to_df.items():
        sheet_df["sheet_name"] = sheet_name
        df = pd.concat([df, sheet_df])
    return df

def get_time_window(df: pd.DataFrame, timestamp: datetime.datetime, hours_before: int = 2, hours_after: int = 2, reset_index: bool = True) -> pd.DataFrame:
    time_from, time_to = (timestamp - datetime.timedelta(hours=hours_before), timestamp + datetime.timedelta(hours=hours_after))
    filtered: pd.DataFrame = df[(df["Datum"] <= time_to) & (df["Datum"] >= time_from)]
    if reset_index:
        filtered = filtered.reset_index(drop=True)
    return filtered

def main():
    df: pd.DataFrame = load_data()
    timestamp: datetime.datetime = datetime.datetime(2024, 8, 27, 12, 0, 0)
    filtered: pd.DataFrame = get_time_window(df, timestamp)
    print(row2str(filtered.iloc[0]))

if __name__ == "__main__":
    main()
