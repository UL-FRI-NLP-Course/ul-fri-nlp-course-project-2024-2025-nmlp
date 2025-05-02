from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)  # lightweight model for semantic similarity

def compute_similarity_score(generated, reference):
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(reference, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

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