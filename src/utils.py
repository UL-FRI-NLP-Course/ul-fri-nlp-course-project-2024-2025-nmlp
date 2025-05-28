import re
import spacy
from spacy.tokens import Doc, Token
from typing import override

POS_WHITELIST: list[str] = ["VERB", "ADJ", "NOUN", "PROPN", "NUM"]
MODEL_SPACY: str = "sl_core_news_lg"

nlp: spacy.language.Language = spacy.load(MODEL_SPACY)

class Paragraph():
    raw: str
    doc: Doc | None
    normalized: str | None
    def __init__(self, raw: str):
        self.raw = raw.strip()
        self.doc = None
        self.normalized = None
        self.normalized_pos = None
    def get_doc(self) -> Doc:
        self.doc = self.doc if self.doc is not None else nlp(self.raw)
        return self.doc
    def get_normalized(self) -> str:
        self.normalized = self.normalized if self.normalized is not None else normalize_str(self.get_doc())
        return self.normalized
    def get_word_count(self) -> int:
        if not self.get_normalized():
            return 0
        return (1 + len(re.findall(r"\s+", self.get_normalized())))
    def get_propn_count(self) -> int:
        doc: Doc = self.get_doc()
        return len(list(token for token in doc if token.pos_ == "PROPN"))
    def get_ne_count(self) -> int:
        doc: Doc = self.get_doc()
        return len(doc.ents)
    @override
    def __repr__(self) -> str:
        return self.__str__()
    @override
    def __str__(self) -> str:
        return f"""
        {self.__class__.__name__}({self.raw})
        """.strip()
    def __eq__(self, other) -> bool:
        if not isinstance(other, Paragraph):
            return False
        return (self.raw == other.raw)
    def __hash__(self) -> int:
        return hash(self.raw)

# Force keep capitalization for proper nouns
def get_lemma_keep_capitalization(token: Token) -> str:
    if token.lemma_ and len(token.lemma_) > 0 and token.pos_ == "PROPN":
        return ((token.lemma_[0].upper() if token.text[0].isupper() else token.lemma_[0]) + token.lemma_[1:])
    return token.lemma_

def normalize_str(text: str | Doc) -> str:
    """
    Input: "Anton Tomaž Linhart je slovenski dramatik, pesnik, zgodovinar in šolnik. Rodil se je 11. decembra 1756 v meščanski družini v Radovljici"
    Output: "Anton Tomaž Linhart slovenski dramatik pesnik zgodovinar šolnik roditi december meščanski družina Radovljica"
    """
    doc: Doc = text if isinstance(text, Doc) else nlp(text)
    return " ".join(get_lemma_keep_capitalization(token) for token in doc if token.pos_.upper() in POS_WHITELIST)
