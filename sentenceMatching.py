from sentence_transformers import SentenceTransformer, util
import re
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # lightweight model for semantic similarity
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