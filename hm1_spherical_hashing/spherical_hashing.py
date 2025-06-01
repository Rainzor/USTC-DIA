import numpy as np
import cupy as cp
from tqdm import tqdm
import scipy.io as io
import os
import matplotlib.pyplot as plt

class SphericalHashing:
    def __init__(self, n_bits, max_iter=100, overlap_ratio=0.25, epsilon_mean=0.1, epsilon_stddev=0.15, num_train_samples=None):
        """Initialize SphericalHashing.
        
        Args:
            n_bits (int): Number of hash bits
            max_iter (int): Maximum number of iterations for training
            overlap_ratio (float): Target overlap ratio between spheres (default: 0.25)
            epsilon_mean (float): Error threshold for mean
            epsilon_stddev (float): Error threshold for standard deviation
            num_train_samples (int): Number of training samples to use (default: all)
        """
        self.n_bits = n_bits
        self.max_iter = max_iter
        self.overlap_ratio = overlap_ratio
        self.epsilon_mean = epsilon_mean
        self.epsilon_stddev = epsilon_stddev
        self.num_train_samples = num_train_samples

        # Sphere Data Structure
        self.centers = None
        self.radii = None
        self.hash_codes = None
        self.features_dim = None
        

    def initialize_spheres(self, features):
        """Initialize spheres with centers near data center"""
        num_samples, self.features_dim = features.shape
        self.centers = cp.zeros((self.n_bits, self.features_dim))
        self.radii = cp.zeros(self.n_bits)
        self.hash_codes = cp.zeros((self.n_bits, num_samples), dtype=bool)


        for i in range(self.n_bits):
            random_indices = cp.random.choice(num_samples, 10, replace=False)
            random_points = features[random_indices]
            self.centers[i] = cp.mean(random_points, axis=0)
    
    def set_radius_hashcode(self, features):
        """Set radius and hash code for each sphere"""
        for i in range(self.n_bits):
            distances = cp.linalg.norm(features - self.centers[i], axis=1)
            self.radii[i] = cp.median(distances)
            self.hash_codes[i] = distances <= self.radii[i]

    def compute_overlaps(self):
        """Compute number of overlaps between all pairs of spheres"""
        overlaps = cp.zeros((self.n_bits, self.n_bits), dtype=cp.int32)
        
        for i in range(self.n_bits):
            for j in range(i+1, self.n_bits):
                overlaps[i, j] = cp.sum(self.hash_codes[i] & self.hash_codes[j])
                overlaps[j, i] = overlaps[i, j]
        overlaps.diagonal()[:] = cp.sum(self.hash_codes, axis=1)
        return overlaps
    
    def fit(self, X, visualize=False):
        """Train the spherical hash model using GPU acceleration.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary hash codes of shape (n_samples, n_bits)
        """
        # Transfer data to GPU
        features = cp.asarray(X)
        num_samples, num_features = features.shape
        
        # Use subset of data for training if specified
        if self.num_train_samples is not None and self.num_train_samples < num_samples:
            train_indices = cp.random.choice(num_samples, self.num_train_samples, replace=False)
            train_features = features[train_indices]
        else:
            train_features = features
            
        # Initialize spheres
        self.initialize_spheres(train_features)

        # Set initial radii and hash codes
        self.set_radius_hashcode(train_features)
        # Compute overlaps
        overlaps = self.compute_overlaps() # [n_bits, n_bits]
        target_overlap = len(train_features) * self.overlap_ratio
        overlap_mean_list = []
        overlap_std_list = []
        # Training iterations
        for iter in tqdm(range(self.max_iter), desc="Training Spherical Hash"):
            # Compute and apply forces
            forces = cp.zeros((self.n_bits, num_features))
            for i in range(self.n_bits-1):
                for j in range(i+1, self.n_bits):
                    alpha = (overlaps[i, j] - target_overlap) / (target_overlap * 2.0)
                    # Compute force between spheres i and j
                    force = alpha * (self.centers[j] - self.centers[i])
                    forces[j] += force
                    forces[i] -= force
            
            # Update sphere centers, radii and hash codes
            self.centers = self.centers + forces / self.n_bits
            self.set_radius_hashcode(train_features)
            
            # Check convergence
            overlaps = self.compute_overlaps()
            overlap_mean = cp.average(abs(overlaps - target_overlap))
            overlap_std = cp.std(overlaps)
            overlap_mean_list.append(cp.asnumpy(overlap_mean)/target_overlap)
            overlap_std_list.append(cp.asnumpy(overlap_std)/target_overlap)
            if (overlap_mean < target_overlap * self.epsilon_mean and 
                overlap_std < target_overlap * self.epsilon_stddev):
                break
            
        if visualize:
            plt.plot(overlap_mean_list, label='overlap_mean', linestyle='-', color='blue', linewidth=2)
            plt.plot(overlap_std_list, label='overlap_std', linestyle='--', color='red', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Overlap')
            plt.title('Overlap Mean and Standard Deviation')
            plt.legend()
            plt.savefig(os.path.join('outputs', 'overlap_mean_and_std.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Generate final hash codes
        hash_codes = cp.zeros((num_samples, self.n_bits), dtype=bool)
        for i in range(self.n_bits):
            distances = cp.linalg.norm(features - self.centers[i], axis=1)
            hash_codes[:, i] = distances <= self.radii[i]
        
        return cp.asnumpy(hash_codes).astype(bool)
    
    def transform(self, X):
        """Generate hash codes for new features using GPU acceleration.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary hash codes of shape (n_samples, n_bits)
        """

        if self.centers is None:
            raise ValueError("Model not trained. Call fit() first.")

        features = cp.asarray(X)
        num_samples = len(features)
        hash_codes = cp.zeros((num_samples, self.n_bits), dtype=bool)
        
        for i in range(self.n_bits):
            distances = cp.linalg.norm(features - self.centers[i], axis=1)
            hash_codes[:, i] = distances <= self.radii[i]
        
        return cp.asnumpy(hash_codes).astype(bool)