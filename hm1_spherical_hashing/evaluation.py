import numpy as np
from scipy.spatial.distance import hamming
from tqdm import tqdm

def calculate_map(query_codes, db_codes, relevance_matrix):
    """计算mAP
    
    Args:
        query_codes: 查询图像的哈希码
        db_codes: 数据库图像的哈希码
        relevance_matrix: 相关性矩阵，shape (n_queries, n_db)
        
    Returns:
        mAP值
    """
    n_queries = len(query_codes)
    aps = []
    
    for i in tqdm(range(n_queries), desc="Computing mAP"):
        # 计算Hamming距离
        distances = np.array([hamming(query_codes[i], db_code) for db_code in db_codes])
        
        # 按距离排序
        sorted_indices = np.argsort(distances)
        
        # 计算AP
        relevant = relevance_matrix[i][sorted_indices]
        ap = 0
        num_relevant = np.sum(relevant)
        
        if num_relevant > 0:
            precision = 0
            for k in range(len(relevant)):
                if relevant[k]:
                    precision += np.sum(relevant[:k+1]) / (k + 1)
            ap = precision / num_relevant
            
        aps.append(ap)
    
    return np.mean(aps)

def calculate_precision_recall(query_codes, db_codes, relevance_matrix, k_values):
    """计算precision@K和recall@K
    
    Args:
        query_codes: 查询图像的哈希码
        db_codes: 数据库图像的哈希码
        relevance_matrix: 相关性矩阵
        k_values: K值列表
        
    Returns:
        precision@K和recall@K的列表
    """
    n_queries = len(query_codes)
    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    
    for i in tqdm(range(n_queries), desc="Computing Precision-Recall"):
        distances = np.array([hamming(query_codes[i], db_code) for db_code in db_codes])
        sorted_indices = np.argsort(distances)
        relevant = relevance_matrix[i][sorted_indices]
        
        for k in k_values:
            if k <= len(relevant):
                precision = np.sum(relevant[:k]) / k
                recall = np.sum(relevant[:k]) / np.sum(relevant)
                precisions[k].append(precision)
                recalls[k].append(recall)
    
    return precisions, recalls 