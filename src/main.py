import pandas as pd

INPUT_DATA_PATH: str = "data/Podatki - PrometnoPorocilo_2022_2023_2024.xlsx"

def main():
    df: pd.DataFrame = pd.read_excel(INPUT_DATA_PATH)

if __name__ == "__main__":
    main()
