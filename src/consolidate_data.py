import re
import spacy
import datetime
import pandas as pd
import Levenshtein
import src.input_data
import src.output_data
import src.utils
from spacy.tokens import Doc
from bs4 import BeautifulSoup
from bs4.element import Tag
from collections import Counter

MODEL_SPACY: str = "sl_core_news_lg"
MIN_MATCHING_WORDS_PER_PARAGRAPH: int = 2
MIN_MATCHING_PARAGRAPHS: int = 2
REGEX_BODY_START: re.Pattern = re.compile(r"^\s*(podatki\s*o\s*promet[u]?|(nujn[ae])?\s*prometn[ae]\s*informacij[ae]\s*)[\.:;]*\s*", flags=re.IGNORECASE)
REGEX_ONLY_CHARS: re.Pattern = re.compile(r"[^a-zčšž]", flags=re.IGNORECASE)

nlp: spacy.language.Language = spacy.load("sl_core_news_lg")

# PROPN – Proper noun: a noun that refers to a specific person, place, or organization, such as “Microsoft” or “John”
# source: https://www.pythonprog.com/spacy-part-of-speech-tags/
# Return True if "enough" proper nouns match, else False
def check_match_propn(str1: str, str2: str) -> bool:
    """
    str1 = "vršič pot cesta ljubljana kot"
    str2 = "predor zamenjati zima slovenija ljubljana"
    matches = 1 (ljubljana)
    total = 3 (vršič, ljubljana, slovenija)
    matches / total = 0.33
    return (0.33 >= 0.5)
    """
    doc1: Doc = nlp(str1)
    doc2: Doc = nlp(str2)
    propn1: set[str] = set(token.text for token in doc1 if token.pos_ == "PROPN")
    propn2: set[str] = set(token.text for token in doc2 if token.pos_ == "PROPN")
    if len(propn1) < 1 or len(propn2) < 1:
        return True
    return (len(propn1 & propn2) >= 0.5 * len(propn1) or len(propn1 & propn2) >= 0.5 * len(propn2))

def check_match_levenshtein(str1: str, str2: str) -> bool:
    str1, str2 = str1.lower(), str2.lower()
    words_1: list[str] = re.split(r"\s+", str1)
    words_2: list[str] = re.split(r"\s+", str2)
    matches_1: int = 0
    matches_2: int = 0
    for word_1 in words_1:
        for word_2 in words_2:
            if Levenshtein.distance(word_1, word_2) < 2:
                matches_1 += 1
                break
    for word_2 in words_2:
        for word_1 in words_1:
            if Levenshtein.distance(word_2, word_1) < 2:
                matches_2 += 1
                break
    return (matches_1/len(words_1) >= 0.3 or matches_2/len(words_2) >= 0.3)

def check_match(str1: str, str2: str) -> bool:
    if not check_match_propn(str1, str2):
        return False
    if not check_match_levenshtein(str1, str2):
        return False
    # TODO: add embeddings comparison here
    return True

def strip_body(text: str) -> str:
    return REGEX_BODY_START.sub("", text)

def remove_unwanted_tag(soup: BeautifulSoup, tag_name: str):
    tags = soup.find(tag_name)
    if tags and hasattr(tags, "children"):
        for tag in tags.children:
            if isinstance(tag, Tag):
                tag.decompose()

def excel_row_to_paragraphs(row: pd.Series) -> tuple[list[str], list[str]]:
    relevant_cols: list[str] = [col for col in row.keys() if col.startswith("Content") and isinstance(row[col], str)]
    relevant_cols = sorted(relevant_cols)
    paragraphs: list[str] = []
    paragraphs_unprocessed: list[str] = []
    for col in relevant_cols:
        soup: BeautifulSoup = BeautifulSoup(str(row[col]), "html.parser")
        remove_unwanted_tag(soup, "a")
        for p in soup.find_all("p"):
            paragraphs.append(src.utils.normalize_str(p.get_text()))
            paragraphs_unprocessed.append(f"<p>{p.get_text()}</p>")
    return paragraphs_unprocessed, paragraphs

def get_matches(str_1: str, str_2: str) -> dict[str, int]:
    """
    Example:
    str1 = "A A B C"
    str2 = "A A C"
    return {"A": 2, "C": 1}
    """
    words_1: list[str] = re.split(r"\s+", str_1)
    words_2: list[str] = re.split(r"\s+", str_2)
    histogram_1: Counter = Counter(words_1)
    histogram_2: Counter = Counter(words_2)
    return {key: min(histogram_1[key], histogram_2[key]) for key in histogram_1.keys() if key in histogram_2.keys()}

def get_paragraph_matches_indexes(paragraphs_rtf: list[str], paragraphs_excel: list[str]) -> list[tuple[int, int]]:
    """
    Example:
    paragraphs_rtf = ["A", "A", "B", "C"]
    paragraphs_excel = ["A", "C", "A"]
    return [(0, 0), (1, 2), (3, 1)]
    """
    matches_indexes: list[tuple[int, int]] = []
    for i, paragraph_rtf in enumerate(paragraphs_rtf):
        most_matched_words: int = 0
        match_indexes: tuple[int, int] = (0, 0)
        for j, paragraph_excel in enumerate(paragraphs_excel):
            if j in map(lambda pair: pair[1], matches_indexes): # Don't map excel paragraph multiple times
                continue
            matches: dict[str, int] = get_matches(paragraph_rtf, paragraph_excel)
            num_matches: int = sum(matches.values())
            if num_matches > most_matched_words:
                most_matched_words = num_matches
                match_indexes = (i, j)
        if most_matched_words >= MIN_MATCHING_WORDS_PER_PARAGRAPH:
            matches_indexes.append(match_indexes)
    return matches_indexes

def main():
    df_rtfs: pd.DataFrame = src.output_data.load_structured()
    df_rtfs["body"] = df_rtfs["body"].apply(strip_body)
    df_excel: pd.DataFrame = src.input_data.load_data()
    io_pairs: list[dict[str, str]] = []
    for _, row_rtf in df_rtfs.iterrows():
        paragraphs_rtf_unprocessed: list[str] = re.split(r"\s*\n+\s*", row_rtf.body)
        paragraphs_rtf = list(map(src.utils.normalize_str, paragraphs_rtf_unprocessed))
        timestamp: datetime.datetime = row_rtf.timestamp.to_pydatetime()
        reports: pd.DataFrame = src.input_data.get_time_window(df_excel, timestamp, hours_before=4, hours_after=1)
        max_matching_paragraphs: int = 0
        input_str, output_str = "", ""
        for _, report in reports.iterrows():
            paragraphs_excel_unprocessed, paragraphs_excel = excel_row_to_paragraphs(report)
            matches_indexes: list[tuple[int, int]] = get_paragraph_matches_indexes(paragraphs_rtf, paragraphs_excel)
            matches_indexes = [(index_rtf, index_excel) for index_rtf, index_excel in matches_indexes if check_match(paragraphs_rtf[index_rtf], paragraphs_excel[index_excel])]
            if len(matches_indexes) > max_matching_paragraphs:
                max_matching_paragraphs = len(matches_indexes)
                input_str = "\n".join(paragraphs_excel_unprocessed[index_excel] for _, index_excel in matches_indexes)
                output_str = row_rtf.header_original + "\n\n" + "\n\n".join(paragraphs_rtf_unprocessed[index_rtf] for index_rtf, _ in matches_indexes)
        if input_str and output_str and max_matching_paragraphs >= MIN_MATCHING_PARAGRAPHS:
            io_pairs.append({"input": input_str, "output": output_str})            
    df_io_pairs: pd.DataFrame = pd.DataFrame(io_pairs)

if __name__ == "__main__":
    main()

