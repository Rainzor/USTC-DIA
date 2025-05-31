import numpy as np
import time
import matplotlib.pyplot as plt
from spherical_hashing import SphericalHashing
from evaluation import calculate_map, calculate_precision_recall

def load_data():
    """加载数据
    这里需要根据实际数据格式进行修改
    """
    # 示例数据加载
    features = np.random.randn(16000, 768)  # 替换为实际数据
    labels = np.random.randint(0, 38, (16000, 5))  # 替换为实际数据
    return features, labels

def create_relevance_matrix(query_labels, db_labels):
    """创建相关性矩阵"""
    n_queries = len(query_labels)
    n_db = len(db_labels)
    relevance = np.zeros((n_queries, n_db))
    
    for i in range(n_queries):
        for j in range(n_db):
            if np.any(np.isin(query_labels[i], db_labels[j])):
                relevance[i, j] = 1
                
    return relevance

def main():
    # 加载数据
    features, labels = load_data()
    
    # 划分查询集和数据库
    query_features = features[:2000]
    query_labels = labels[:2000]
    db_features = features[2000:]
    db_labels = labels[2000:]
    
    # 创建相关性矩阵
    relevance_matrix = create_relevance_matrix(query_labels, db_labels)
    
    # 测试不同的比特数
    bit_lengths = [16, 32, 64, 128]
    results = {}
    
    for n_bits in bit_lengths:
        print(f"\nTesting {n_bits} bits...")
        
        # 训练球面哈希
        hasher = SphericalHashing(n_bits)
        hasher.fit(db_features)
        
        # 转换特征为哈希码
        start_time = time.time()
        db_codes = hasher.transform(db_features)
        query_codes = hasher.transform(query_features)
        encoding_time = time.time() - start_time
        
        # 计算存储消耗
        storage = db_codes.nbytes / 1024  # KB
        
        # 计算检索时间
        start_time = time.time()
        for i in range(len(query_codes)):
            _ = [hamming(query_codes[i], db_code) for db_code in db_codes]
        retrieval_time = (time.time() - start_time) / len(query_codes)
        
        # 计算mAP
        map_score = calculate_map(query_codes, db_codes, relevance_matrix)
        
        # 计算precision@K和recall@K
        k_values = [1, 5, 10, 20, 50, 100]
        precisions, recalls = calculate_precision_recall(
            query_codes, db_codes, relevance_matrix, k_values)
        
        results[n_bits] = {
            'mAP': map_score,
            'storage': storage,
            'retrieval_time': retrieval_time,
            'precisions': precisions,
            'recalls': recalls
        }
        
        print(f"mAP: {map_score:.4f}")
        print(f"Storage: {storage:.2f} KB")
        print(f"Average retrieval time: {retrieval_time*1000:.2f} ms")
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # Precision@K曲线
    plt.subplot(131)
    for n_bits in bit_lengths:
        precisions = [np.mean(results[n_bits]['precisions'][k]) for k in k_values]
        plt.plot(k_values, precisions, label=f'{n_bits} bits')
    plt.xlabel('K')
    plt.ylabel('Precision@K')
    plt.legend()
    
    # Recall@K曲线
    plt.subplot(132)
    for n_bits in bit_lengths:
        recalls = [np.mean(results[n_bits]['recalls'][k]) for k in k_values]
        plt.plot(k_values, recalls, label=f'{n_bits} bits')
    plt.xlabel('K')
    plt.ylabel('Recall@K')
    plt.legend()
    
    # 性能指标对比
    plt.subplot(133)
    x = np.arange(len(bit_lengths))
    width = 0.35
    
    maps = [results[n_bits]['mAP'] for n_bits in bit_lengths]
    plt.bar(x, maps, width, label='mAP')
    plt.xlabel('Bit Length')
    plt.ylabel('mAP')
    plt.xticks(x, bit_lengths)
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()

if __name__ == "__main__":
    main() 