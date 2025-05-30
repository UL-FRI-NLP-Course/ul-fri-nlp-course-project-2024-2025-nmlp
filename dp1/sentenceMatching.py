from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)  # lightweight model for semantic similarity

# cosine similarity function
def compute_similarity_score(generated, reference):
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(reference, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# gruping based on semantic similarity
def group_unique_semantic(series, similarity_threshold=0.7):
    """
    Groups sentences based on semantic similarity.
    """
    # Split all texts into sentences
    all_sentences = []
    for text in series:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    sentences = [s.strip() for s in all_sentences if s.strip()]
    

    if not sentences:
        return ""

    # Get embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Track which sentences are unique
    unique_sentences = []
    used = set()
    
    for i, emb in enumerate(embeddings):
        if i in used:
            continue
        similar = util.cos_sim(emb, embeddings)[0]
        indices = [j for j, score in enumerate(similar) if score > similarity_threshold and j != i]
        used.update(indices)
        unique_sentences.append(sentences[i])  # Keep the first one
        
    return " ".join(unique_sentences)

# grouping based on semantic similarity and selecting the most informative sentence
def group_unique_semantic_informative(series, similarity_threshold=0.7):
    """
    Groups sentences based on semantic similarity and 
    selects the most informative one from each group.
    """
    # Flatten into a list of sentences
    all_sentences = []
    for text in series:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    sentences = [s.strip() for s in all_sentences if s.strip()]
    
    if not sentences:
        return ""

    # Encode
    embeddings = model.encode(sentences, convert_to_tensor=True)
    n = len(sentences)
    used = set()
    keep = []

    for i in range(n):
        if i in used:
            continue
        # Find all similar sentences
        sims = util.cos_sim(embeddings[i], embeddings)[0]
        group = [j for j in range(n) if sims[j] > similarity_threshold]
        best_idx = max(group, key=lambda j: len(sentences[j]))  # Use length as proxy for informativeness
        keep.append(sentences[best_idx])
        used.update(group)

    return " ".join(keep)

# grouping based on TF-IDF scores and semantic similarity
def group_tf_idf_informative(series, similarity_threshold=0.7):
    # Step 1: Flatten and clean
    all_sentences = []
    for text in series:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    sentences = [s.strip() for s in all_sentences if s.strip()]
    if not sentences:
        return ""

    # Step 2: Embed
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Step 3: TF-IDF
    tfidf = TfidfVectorizer().fit(sentences)
    scores = tfidf.transform(sentences).sum(axis=1).A1

    # Step 4: Group & Select
    used = set()
    result = []
    for i in range(len(sentences)):
        if i in used:
            continue
        sims = util.cos_sim(embeddings[i], embeddings)[0]
        group = [j for j in range(len(sentences)) if sims[j] > similarity_threshold]
        best_idx = max(group, key=lambda j: scores[j])
        result.append(sentences[best_idx])
        used.update(group)
    return " ".join(result)

# TODO: add more keywords to the list
road_keywords = [
    "avtocesta", "hitra cesta", "glavna cesta", "regionalna cesta", "Ljubljana", "Maribor",
    "Karavanke", "Koper", "Obrežje", "Dragučova", "Slivnica", "Gorenjska", "Štajerska",
    "Dolenjska", "Pomurska", "Podravska", "Razcep", "uvoz", "izvoz", "zaprta", "zastoj",
    "obvoznica", "delna zapora", "popravilo", "vzdrževanje", "dela na cesti",
    "prometno obvestilo", "prometno poročilo", "prometne informacije"
]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Group sentences with a preference for named entities related to roads
def group_with_named_entity_preference(series, similarity_threshold=0.7):
    all_sentences = []
    for text in series:
        all_sentences.extend(re.split(r'(?<=[.!?])\s+', text))
    sentences = [s.strip() for s in all_sentences if s.strip()]
    if not sentences:
        return ""

    embeddings = model.encode(sentences, convert_to_tensor=True)

    def priority_score(sentence):
        return sum(1 for kw in road_keywords if kw.lower() in sentence.lower())

    used = set()
    result = []
    for i in range(len(sentences)):
        if i in used:
            continue
        sims = util.cos_sim(embeddings[i], embeddings)[0]
        group = [j for j in range(len(sentences)) if sims[j] > similarity_threshold]
        best_idx = max(group, key=lambda j: priority_score(sentences[j]))
        result.append(sentences[best_idx])
        used.update(group)
    return " ".join(result)

# sentenceMatching.py – Example usages (was mostly used for: extract.py)
# -------------------------------------------------------------

# compute_similarity_score:
# Computes cosine similarity between two texts using sentence embeddings.
# Useful in extract.py's `test_preprocessing_strategies()` to evaluate how close the generated prompt is to the RTF summary.
# compute_similarity_score(
#     generated="Prometne informacije ...",
#     reference="Dežurna služba poroča ..."
# )

# group_unique_semantic:
# Groups semantically similar sentences and keeps one representative from each group.
# Used inside extract.py as a grouping function in `create_prompt_input()` or `prepare_prompt_from_datetime()`.
# group_unique_semantic(pd.Series([
#     "Zastoji na AC A1 med Celjem in Dravskim Poljem.",
#     "Na avtocesti A1 so zastoji pri Celju.",
#     "Počasna vožnja v bližini Dravskega Polja."
# ]))

# group_unique_semantic_informative:
# Same as `group_unique_semantic`, but chooses the *most informative* (longest) sentence per group.
# Used as a default grouping strategy in many extract.py functions like `prepare_prompt_from_datetime()`.
# group_unique_semantic_informative(pd.Series([
#     "Zastoj na AC A1 med Celjem in Dravskim Poljem.",
#     "Gneča na avtocesti v bližini Celja.",
#     "Na cesti med Dravskim Poljem in Celjem je počasna vožnja zaradi del."
# ]))

# group_tf_idf_informative:
# Uses TF-IDF to score informativeness, and semantic similarity to group.
# Useful in extract.py to test alternative prompt-generation strategies.
# group_tf_idf_informative(pd.Series([
#     "Počasna vožnja na AC A2.",
#     "Zaradi nesreče je promet upočasnjen.",
#     "Zastoj na avtocesti A2 med Ljubljano in Kranjem."
# ]))

# group_with_named_entity_preference:
# Prefers sentences containing road-related keywords (locations, objects) when choosing group representatives.
# Used in extract.py’s `test_preprocessing_strategies()` to evaluate grouping strategies with domain-specific bias.
# group_with_named_entity_preference(pd.Series([
#     "Zastoj na uvozu za Štajersko avtocesto.",
#     "Gneča pred razcepom Dragučova.",
#     "Zaradi nesreče pri izvozu Ljubljana vzhod je promet upočasnjen."
# ]))