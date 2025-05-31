import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

class SphericalHashing:
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.centers = None
        self.radii = None
        
    def fit(self, X):
        """训练球面哈希函数
        
        Args:
            X: 特征向量矩阵，shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 随机选择n_bits个中心点
        indices = np.random.choice(n_samples, self.n_bits, replace=False)
        self.centers = X[indices]
        
        # 计算每个中心点到所有样本的距离
        distances = cdist(self.centers, X)
        
        # 计算每个超球面的半径
        self.radii = np.median(distances, axis=1)
        
    def transform(self, X):
        """将特征向量转换为二值哈希码
        
        Args:
            X: 特征向量矩阵，shape (n_samples, n_features)
            
        Returns:
            二值哈希码矩阵，shape (n_samples, n_bits)
        """
        distances = cdist(X, self.centers)
        binary_codes = (distances <= self.radii).astype(np.int32)
        return binary_codes 