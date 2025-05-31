import numpy as np
from sklearn.metrics import average_precision_score
import time

def hamming_distance(code0, code2):
    """Calculate Hamming distance between two binary codes"""
    # the smaller the distance, the more similar
    return np.sum(code0 != code2, axis=1)

def normalize_hamming_distance(code0, code2):
    """Normalize Hamming distance"""
    return hamming_distance(code0, code2) / (np.sum(code0 & code2, axis=1)+ 1)

def calculate_map_and_curves(query_codes, db_codes, relevance_matrix):
    """Calculate mAP, recall, precision"""
    n_queries = len(query_codes)
    ap_scores = []
    recall_scores = []
    precision_scores = []
    
    for i in range(n_queries):
        # Calculate Hamming distances
        distances = hamming_distance(query_codes[i], db_codes)
        # Sort by distance
        sorted_indices = np.argsort(distances)
        # Get relevance scores for sorted results
        relevant = relevance_matrix[i][sorted_indices]

        ap = average_precision_score(relevant, -distances)
        ap_scores.append(ap)
        # cumulative sum of relevant items divided by the total number of relevant items
        recall = np.cumsum(relevant) / np.sum(relevant)
        # cumulative sum of relevant items divided by the number of items considered
        precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1) 
        recall_scores.append(recall)
        precision_scores.append(precision)
    
    return np.mean(ap_scores, axis=0), np.mean(recall_scores, axis=0), np.mean(precision_scores, axis=0)

def evaluate_storage_and_time(query_codes, db_codes):
    database_storage = query_codes.nbytes + db_codes.nbytes
    start_time = time.time()
    for query_code in query_codes:
        _ = hamming_distance(query_code, db_codes)
    query_time = (time.time() - start_time) / len(query_codes)
    return database_storage, query_time