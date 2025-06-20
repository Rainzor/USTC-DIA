import numpy as np
from sklearn.metrics import average_precision_score
import time

def hamming_distance(code0, code2):
    """Calculate Hamming distance between two binary codes"""
    # the smaller the distance, the more similar
    return np.sum(code0 != code2, axis=1)

def spherical_hamming_distance(code0, code2):
    """Normalize Hamming distance"""
    return np.sum(code0 ^ code2, axis=1) / (np.sum(code0 & code2, axis=1)+ 0.1)

def calculate_map_at_k(query_codes, db_codes, relevance_matrix, k=None):
    """Calculate mAP@K for a specific K value"""
    n_queries = len(query_codes)
    ap_scores = []
    
    for i in range(n_queries):
        # Calculate Hamming distances
        distances = spherical_hamming_distance(query_codes[i], db_codes)
        # Sort by distance
        sorted_indices = np.argsort(distances)
        # Get relevance scores for sorted results
        relevant = relevance_matrix[i][sorted_indices]
        
        if k is not None:
            # Only consider top K results
            relevant = relevant[:k]
        
        precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1) 
        ap = np.sum(precision*relevant) / np.sum(relevant)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def calculate_map_and_curves(query_codes, db_codes, relevance_matrix):
    """Calculate mAP, recall, precision"""
    n_queries = len(query_codes)
    ap_scores = []
    recall_scores = []
    precision_scores = []
    
    for i in range(n_queries):
        # Calculate Hamming distances
        distances = spherical_hamming_distance(query_codes[i], db_codes)
        # Sort by distance
        sorted_indices = np.argsort(distances)
        # Get relevance scores for sorted results
        relevant = relevance_matrix[i][sorted_indices]

        # ap = average_precision_score(relevant, -distances)
        # ap_scores.append(ap)
        # cumulative sum of relevant items divided by the total number of relevant items
        recall = np.cumsum(relevant) / np.sum(relevant)
        # cumulative sum of relevant items divided by the number of items considered
        precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1) 
        recall_scores.append(recall)
        precision_scores.append(precision)
        ap = np.sum(precision*relevant) / np.sum(relevant)
        ap_scores.append(ap)

    m_precision = np.mean(precision_scores, axis=0)
    m_recall = np.mean(recall_scores, axis=0)
    m_ap = np.mean(ap_scores)
    return m_ap, m_recall, m_precision

def evaluate_storage_and_time(query_codes, db_codes):
    database_storage = query_codes.nbytes + db_codes.nbytes
    start_time = time.time()
    for query_code in query_codes:
        _ = spherical_hamming_distance(query_code, db_codes)
    query_time = (time.time() - start_time) / len(query_codes)
    return database_storage, query_time