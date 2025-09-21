def precision_at_k(relevant_ids, retrieved_ids, k=5):
    rset = set(relevant_ids)
    topk = retrieved_ids[:k]
    return sum(1 for x in topk if x in rset) / max(1, min(k, len(retrieved_ids)))