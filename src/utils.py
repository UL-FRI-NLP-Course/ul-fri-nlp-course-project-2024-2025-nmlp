import spacy
from spacy.tokens import Doc, Token

POS_WHITELIST: list[str] = ["VERB", "ADJ", "NOUN", "PROPN"]

nlp: spacy.language.Language = spacy.load("sl_core_news_lg")

# Force keep capitalization for proper nouns
def get_lemma_keep_capitalization(token: Token) -> str:
    if token.lemma_ and len(token.lemma_) > 0 and token.pos_ == "PROPN":
        return token.lemma_[0].upper() + token.lemma_[1:]
    return token.lemma_

def normalize_str(text: str) -> str:
    doc: Doc = nlp(text)
    return " ".join(get_lemma_keep_capitalization(token) for token in doc if token.pos_.upper() in POS_WHITELIST)
