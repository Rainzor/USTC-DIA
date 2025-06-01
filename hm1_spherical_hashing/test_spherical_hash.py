import numpy as np
import cupy as cp
import time
from spherical_hashing import SphericalHashing
import matplotlib.pyplot as plt
from tqdm import tqdm
from evaluation import calculate_map_and_curves, evaluate_storage_and_time

def test_spherical_hash():
    """测试球面哈希算法的检索性能"""
    
    # 1. 加载数据
    print("Loading dataset...")
    data = np.load('datasets/data.npz')
    features = data['arr_0']  # 16000x768 的图像特征
    labels = data['arr_1']    # 16000x38 的图像标签
    
    # 2. 分割数据集
    data_size = len(features)
    n_query = 1000
    query_features = features[:n_query]
    query_labels = labels[:n_query].astype(bool)
    db_features = features[n_query:]
    db_labels = labels[n_query:].astype(bool)
    
    # 3. 创建相关性矩阵
    print("Creating relevance matrix...")
    relevance_matrix = np.zeros((n_query, len(db_features)), dtype=bool)
    for i in range(n_query):
        for j in range(len(db_features)):
            # 如果两个图像共享任何标签，则认为它们是相关的
            relevance_matrix[i, j] = np.any(db_labels[j] & query_labels[i]).astype(bool)
    
    # 4. 训练哈希模型
    n_bits = 64
    print(f"\nTraining with {n_bits} bits...")
    hasher = SphericalHashing(n_bits=n_bits, max_iter=200, overlap_ratio=0.25, epsilon_mean=0.01, epsilon_stddev=0.01)
    
    # 训练并记录时间
    start_time = time.time()
    db_codes = hasher.fit(db_features, visualize=True)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # 5. 生成查询集的哈希码
    query_codes = hasher.transform(query_features)
    
    # 6. 计算 mAP, recall, precision
    mAP, recall, precision = calculate_map_and_curves(query_codes, db_codes, relevance_matrix)    

    # 7. 计算存储消耗和检索时间
    storage, query_time = evaluate_storage_and_time(query_codes, db_codes)
    
    # 8. 输出结果
    print(f"mAP: {mAP:.4f}")
    print(f"Storage Consumption: {storage / 1024:.2f} KB")
    print(f"Average Retrieval Time: {query_time:.4f} seconds")

    # 9. 绘制precision-recall曲线
    plt.plot(recall, precision, label=f'{n_bits}-bit')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = test_spherical_hash() 