from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)  # lightweight model for semantic similarity
# model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def eval(function, ground_truth, generated_output):
    """
    Evaluates the generated output against the ground truth using the provided function.
    
    Args:
        function: The evaluation function to use.
        ground_truth: The ground truth data.
        generated_output: The generated output to evaluate.
        
    Returns:
        The evaluation result.
    """
    if not callable(function):
        raise ValueError("Provided function is not callable.")
    
    return function(ground_truth, generated_output)

def eval_multiple(functions, ground_truth, generated_output):
    """
    Evaluates the generated output against the ground truth using multiple functions.
    
    Args:
        functions: A list of evaluation functions to use.
        ground_truth: The ground truth data.
        generated_output: The generated output to evaluate.
        
    Returns:
        A dictionary with function names as keys and evaluation results as values.
    """
    if not all(callable(func) for func in functions):
        raise ValueError("All provided functions must be callable.")
    
    results = {}
    for func in functions:
        results[func.__name__] = eval(func, ground_truth, generated_output)
    
    return results


def f1_token_overlap(gt, pred):
    """
    Computes F1 score based on word overlap between two texts.

    Args:
        gt (str): Ground truth text.
        pred (str): Generated output text.

    Returns:
        float: F1 score based on token overlap.
    """
    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())
    common = gt_tokens & pred_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Lexical Similarities (Exact/Token-level):
def jaccard_similarity(ground_truth, generated_output):
    """
    Computes the Jaccard similarity between two strings.

    Args:
        a (str): The first string.
        b (str): The second string.
    """
    if not isinstance(ground_truth, str) or not isinstance(generated_output, str):
        raise ValueError("Both inputs must be strings.")
    
    a, b = ground_truth, generated_output
    if not a or not b:
        return 0.0  # Return 0 if either string is empty
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b)

def bleu_score(ground_truth, generated_output):
    """
    Computes the BLEU score between two strings.

    Args:
        ground_truth (str): The reference string.
        generated_output (str): The generated string.
        
    Returns:
        float: The BLEU score.
    """
    if not isinstance(ground_truth, str) or not isinstance(generated_output, str):
        raise ValueError("Both inputs must be strings.")
    
    reference = [ground_truth.split()]
    candidate = generated_output.split()
    return sentence_bleu(reference, candidate)

def rouge_l_score(ground_truth, generated_output):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ground_truth, generated_output)['rougeL'].fmeasure

# Character-level Similarity

def levenshtein_ratio(ground_truth, generated_output):
    return Levenshtein.ratio(ground_truth, generated_output)

# Embedding-based Semantic Similarity
def embedding_similarity(ground_truth, generated_output):
    embeddings = model.encode([ground_truth, generated_output])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Precision
def precision_tokens(ground_truth, generated_output):
    gt_tokens, pred_tokens = set(ground_truth.split()), set(generated_output.split())
    tp = len(gt_tokens & pred_tokens)
    return tp / len(pred_tokens) if pred_tokens else 0.0

# Recall
def recall_tokens(ground_truth, generated_output):
    gt_tokens, pred_tokens = set(ground_truth.split()), set(generated_output.split())
    tp = len(gt_tokens & pred_tokens)
    return tp / len(gt_tokens) if gt_tokens else 0.0


# Example usage:
functions = [
    f1_token_overlap,
    jaccard_similarity,
    bleu_score,
    rouge_l_score,
    levenshtein_ratio,
    embedding_similarity,
    precision_tokens,
    recall_tokens
]
results = eval_multiple(functions, "This is a test sentence.", "This is a test sentence for evaluation.")
print("Evaluation Results:")
for func_name, score in results.items():
    print(f"{func_name}: {score:.4f}")

