import numpy as np

def precision_at_k(actual: list, predicted: list, k: int = 10) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    relevant_retrieved = len(set(predicted_k).intersection(set(actual)))
    return relevant_retrieved / k

def recall_at_k(actual: list, predicted: list, k: int = 10) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    relevant_retrieved = len(set(predicted_k).intersection(set(actual)))
    return relevant_retrieved / len(actual)

def mrr(actual: list, predicted: list) -> float:
    for i, p in enumerate(predicted):
        if p in actual:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(actual: list, predicted: list) -> float:
    if not actual:
        return 0.0
    ap = 0.0
    relevant_hits = 0
    for i, p in enumerate(predicted):
        if p in actual:
            relevant_hits += 1
            ap += relevant_hits / (i + 1)
    return ap / len(actual)

def dcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    predicted_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted_k):
        if p in actual:
            # Assuming relevance is binary (1) for this calculation
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def ndcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    if idcg == 0:
        return 0.0
    return dcg_at_k(actual, predicted, k) / idcg

if __name__ == "__main__":
    # Quick sanity check test
    actual = ['c1', 'c2', 'c3']
    predicted = ['c4', 'c1', 'c5', 'c2']
    print(f"P@3: {precision_at_k(actual, predicted, 3):.2f}")
    print(f"R@3: {recall_at_k(actual, predicted, 3):.2f}")
    print(f"MRR: {mrr(actual, predicted):.2f}")
    print(f"MAP: {average_precision(actual, predicted):.2f}")
    print(f"NDCG@4: {ndcg_at_k(actual, predicted, 4):.2f}")
