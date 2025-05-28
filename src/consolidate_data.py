import re
import json
import spacy
import datetime
import pandas as pd
import Levenshtein
import threading
import src.input_data
import src.output_data
import src.utils
from src.output_data import OutputParagraph, OutputReport
from src.input_data import InputParagraph, InputReport
from spacy.tokens import Doc
from collections import Counter
from typing import Any, Iterator, override
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

MODEL_ST: str = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_MATCHING_WORDS_PER_PARAGRAPH: int = 2
MIN_MATCHING_PARAGRAPHS: int = 2
REGEX_BODY_START: re.Pattern = re.compile(r"^\s*(podatki\s*o\s*promet[u]?|(nujn[ae])?\s*prometn[ae]\s*informacij[ae]\s*)[\.:;]*\s*", flags=re.IGNORECASE)
REGEX_ONLY_CHARS: re.Pattern = re.compile(r"[^a-zčšž]", flags=re.IGNORECASE)
OUTPUT_PATH: str = "./train.jsonl"
MAX_WORKERS: int = 16

nlp: spacy.language.Language = spacy.load(src.utils.MODEL_SPACY)
embeddings_model: SentenceTransformer = SentenceTransformer(MODEL_ST)

class IOParagraph():
    par_in: InputParagraph
    par_out: OutputParagraph
    def __init__(self, par_in: InputParagraph, par_out: OutputParagraph):
        self.par_in = par_in
        self.par_out = par_out
    @override
    def __repr__(self) -> str:
        return self.__str__()
    @override
    def __str__(self) -> str:
        return f"IOParagraph(\n\t{self.par_in}\n\t{self.par_out}\n)"
    def __iter__(self) -> Iterator[tuple[str, str]]:
        yield "in", self.par_in.raw
        yield "out", self.par_out.raw

class MatchStats():
    par_in: InputParagraph
    par_out: OutputParagraph
    matching_words: int
    matching_proper_nouns: int
    matching_named_entities: int
    similarity_score: float
    def __init__(self, par_in: InputParagraph, par_out: OutputParagraph, matching_words: int, matching_proper_nouns: int, matching_named_entities: int, similarity_score: float) -> None:
        self.par_in = par_in
        self.par_out = par_out
        self.matching_words = matching_words
        self.matching_proper_nouns = matching_proper_nouns
        self.matching_named_entities = matching_named_entities
        self.similarity_score = similarity_score
    def __eq__(self, other) -> bool:
        if not isinstance(other, MatchStats):
            return False
        return ((self.matching_words == other.matching_words) and (self.matching_proper_nouns == other.matching_proper_nouns) and (self.matching_named_entities == other.matching_named_entities) and (self.similarity_score == other.similarity_score))
    def __lt__(self, other):
        if not isinstance(other, MatchStats):
            return False
        diff_matching_propn: int = abs(self.matching_proper_nouns - other.matching_proper_nouns)
        diff_matching_ne: int = abs(self.matching_named_entities - other.matching_named_entities)
        diff_similarity_score: float = abs(self.similarity_score - other.similarity_score)
        if self.matching_words < other.matching_words and diff_matching_propn < 3 and diff_matching_ne < 3 and diff_similarity_score < 0.3:
            return True
        if self.matching_proper_nouns < other.matching_proper_nouns and diff_matching_ne < 3 and diff_similarity_score < 0.3:
            return True
        if self.matching_named_entities < other.matching_named_entities and diff_similarity_score < 0.3:
            return True
        return (self.similarity_score < other.similarity_score)
    def is_match(self) -> bool:
        # At least this fraction of words from one paragraph must have corresponding word in the other paragraph
        if self.matching_words >= 0.4 * min(self.par_out.get_word_count(), self.par_in.get_word_count()):
            return True
        if self.matching_words < 0.15 * min(self.par_out.get_word_count(), self.par_in.get_word_count()):
            return False
        # At least some number of proper nouns must match AND a fraction of both paragraphs' proper nouns as well
        if self.matching_proper_nouns >= max(2, 0.4 * max(self.par_in.get_propn_count(), self.par_out.get_propn_count())):
            return True
        if self.matching_proper_nouns < 0.51 * (max(self.par_in.get_propn_count(), self.par_out.get_propn_count()) - 2):
            return False
        if abs(self.par_in.get_propn_count() - self.par_out.get_propn_count()) > 2:
            return False
        if self.matching_named_entities >= max(2, 0.4 * max(self.par_in.get_ne_count(), self.par_out.get_ne_count())):
            return True
        if self.matching_named_entities < 0.51 * (max(self.par_in.get_ne_count(), self.par_out.get_ne_count()) - 2):
            return False
        if abs(self.par_in.get_ne_count() - self.par_out.get_ne_count()) > 2:
            return False
        if 0.6 < self.similarity_score < 0.7:
            print(f"Warning: don't know if this is a match or not\n{self}")
        return (self.similarity_score > 0.70)
    @override
    def __repr__(self) -> str:
        return self.__str__()
    @override
    def __str__(self) -> str:
        return f"""
MatchStats(
    par_in: {self.par_in}
    par_out: {self.par_out}
    matching_words: {self.matching_words}
    matching_proper_nouns: {self.matching_proper_nouns}
    matching_named_entities: {self.matching_named_entities}
    similarity_score: {self.similarity_score}
)
        """.strip()

def count_matches_levenshtein(str1: str, str2: str, max_distance: int = 1) -> int:
    str1, str2 = str1.lower(), str2.lower()
    words_1: list[str] = re.split(r"\s+", str1)
    words_2: list[str] = re.split(r"\s+", str2)
    indexes_used: list[int] = []
    matches: int = 0
    for word_1 in words_1:
        for j, word_2 in enumerate(words_2):
            if j in indexes_used:
                continue
            if Levenshtein.distance(word_1, word_2) <= max_distance:
                matches += 1
                indexes_used.append(j)
    return matches

def count_matches_propn(str1: str | Doc, str2: str | Doc) -> int:
    doc1: Doc = str1 if isinstance(str1, Doc) else nlp(str1)
    doc2: Doc = str2 if isinstance(str2, Doc) else nlp(str2)
    propn1: list[str] = [token.lemma_.lower() for token in doc1 if token.pos_ == "PROPN"]
    propn2: list[str] = [token.lemma_.lower() for token in doc2 if token.pos_ == "PROPN"]
    # c1: Counter = Counter(propn1)
    # c2: Counter = Counter(propn2)
    # return sum(min(c1[k], c2[k]) for k in c1.keys() if k in c2.keys())
    return count_matches_levenshtein(" ".join(propn1), " ".join(propn2))

def count_matches_ne(str1: str | Doc, str2: str | Doc) -> int:
    doc1: Doc = str1 if isinstance(str1, Doc) else nlp(str1)
    doc2: Doc = str2 if isinstance(str2, Doc) else nlp(str2)
    ne1: list[str] = [token.lemma_.lower() for token in doc1 if token.ent_type_]
    ne2: list[str] = [token.lemma_.lower() for token in doc2 if token.ent_type_]
    return count_matches_levenshtein(" ".join(ne1), " ".join(ne2))

def get_match_stats(par_in: InputParagraph, par_out: OutputParagraph) -> MatchStats:
    matching_words: int = count_matches_levenshtein(par_in.get_normalized(), par_out.get_normalized())
    matching_propn: int = count_matches_propn(par_in.get_doc(), par_out.get_doc())
    matching_ne: int = count_matches_ne(par_in.get_doc(), par_out.get_doc())
    similarity_score: float = embeddings_model.similarity(par_in.get_embedding(embeddings_model), par_out.get_embedding(embeddings_model)).item()
    return MatchStats(par_in, par_out, matching_words, matching_propn, matching_ne, similarity_score)

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

def ensure_punctuation(s: str) -> str:
    if not isinstance(s, str) or not s or len(s) < 1:
        return s
    if s[-1] not in [".", "!", "?"]:
        s += "."
    return s

def remove_duplicates_in_list(lst: list[Any]) -> list[Any]:
    lst_new: list[Any] = []
    for x in lst:
        if x not in lst_new:
            lst_new.append(x)
    return lst_new

def get_io_pairs(rtf: OutputReport, excels: list[InputReport]) -> list[IOParagraph]:
    pars_in: list[InputParagraph] = remove_duplicates_in_list([par for excel in excels for par in excel.paragraphs])
    io_pairs: list[IOParagraph] = []
    for par_out in rtf.paragraphs:
        best_match: MatchStats = get_match_stats(pars_in[0], par_out)
        for par_in in pars_in[1:]:
            match_stats: MatchStats = get_match_stats(par_in, par_out)
            if match_stats > best_match:
                best_match = match_stats
        if best_match.is_match():
            io_pairs.append(IOParagraph(best_match.par_in, best_match.par_out))
    return io_pairs

def main():
    df_rtfs: pd.DataFrame = src.output_data.load_structured()
    df_rtfs["body"] = df_rtfs["body"].apply(strip_body)
    df_excel: pd.DataFrame = src.input_data.load_data()
    df_rtfs = df_rtfs.iloc[:100]
    lock: threading.Lock = threading.Lock()
    counter: int = 0
    total: int = len(df_rtfs)
    with open(OUTPUT_PATH, "wt") as file:
        def process_rtf(tup: tuple[int, pd.Series]):
            _, row_rtf = tup
            rtf: OutputReport = OutputReport(row_rtf)
            timestamp: datetime.datetime = row_rtf.timestamp.to_pydatetime()
            df_excel_subset: pd.DataFrame = src.input_data.get_time_window(df_excel, timestamp, hours_before=4, hours_after=1)
            excels: list[InputReport] = list(InputReport(row_excel) for _, row_excel in df_excel_subset.iterrows())
            io_pairs_new: list[IOParagraph] = get_io_pairs(rtf, excels)
            input_joined: str = " ".join(ensure_punctuation(io_pair.par_in.raw) for io_pair in io_pairs_new)
            output_joined: str = " ".join(ensure_punctuation(io_pair.par_out.raw) for io_pair in io_pairs_new)
            nonlocal counter
            with lock:
                if len(io_pairs_new) > 0:
                    file.write(json.dumps({"vhod": input_joined, "izhod": output_joined}, ensure_ascii=False) + "\n")
                    file.flush()
                counter += 1
                print(f"[{counter}/{total}]")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(process_rtf, df_rtfs.iterrows())

if __name__ == "__main__":
    main()

