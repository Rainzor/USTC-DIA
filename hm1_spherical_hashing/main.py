import numpy as np
import time
import matplotlib.pyplot as plt
from spherical_hashing import SphericalHashing
from evaluation import calculate_map_and_curves, evaluate_storage_and_time, calculate_map_at_k
import os

def load_data():
    """Load the dataset from npz file"""
    print("Loading dataset...")
    data = np.load('datasets/data.npz')
    features = data['arr_0']  # 16000x768 image features
    labels = data['arr_1']    # 16000x38 image labels
    return features, labels


def main():
    # Load dataset
    features, labels = load_data()

    # Hyper-parameters
    data_size = len(features)
    # data_size = 4000
    iteration_times = 200
    epsilon_mean = 0.1
    epsilon_stddev = 0.15
    bit_lengths = [16, 32, 64, 128]
    
    # Split into query and database sets
    query_features = features[:1000]
    query_labels = labels[:1000].astype(bool)
    db_features = features[1000: data_size]
    db_labels = labels[1000:data_size].astype(bool)
    
    # Create relevance matrix
    print("Creating relevance matrix...")
    n_queries = len(query_labels)
    n_db = len(db_labels)
    relevance_matrix = np.zeros((n_queries, n_db), dtype=bool)
    for i in range(n_queries):
        for j in range(n_db):
            # If any label is shared, then the two images are relevant
            relevance_matrix[i, j] = np.any(query_labels[i] & db_labels[j]) 
    

    mAP_list = []
    mAP_at_100_list = []
    storage_list = []
    query_time_list = []
    recall_list = {}
    precision_list = {}

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for n_bits in bit_lengths:
        print(f"\nTesting {n_bits} bits...")
        
        # Initialize and train SphericalHashing
        hasher = SphericalHashing(n_bits=n_bits, max_iter=iteration_times, epsilon_mean=epsilon_mean, epsilon_stddev=epsilon_stddev)
        
        # Train the hasher
        db_codes = hasher.fit(db_features)
        
        # Transform the query features
        query_codes = hasher.transform(query_features)

        # Calculate mAP, recall, precision
        mAP, recall, precision = calculate_map_and_curves(query_codes, db_codes, relevance_matrix)
        
        # Calculate mAP@100
        mAP_at_100 = calculate_map_at_k(query_codes, db_codes, relevance_matrix, k=100)
        
        # Calculate storage and time
        storage, query_time = evaluate_storage_and_time(query_codes, db_codes)
        
        mAP_list.append(mAP)
        mAP_at_100_list.append(mAP_at_100)
        storage_list.append(storage)
        query_time_list.append(query_time)
        recall_list[n_bits] = recall
        precision_list[n_bits] = precision
        print(f"n_bits: {n_bits}, mAP: {mAP:.4f}, mAP@100: {mAP_at_100:.4f}, storage: {storage/1024:.2f} KB, query_time: {query_time:.4f} s")
    
    # Create mAP vs Bits plot
    plt.figure(figsize=(10, 6))
    plt.plot(bit_lengths, mAP_list, marker='o', linestyle='-', color='#2E86C1', linewidth=2, markersize=8, label='mAP')
    plt.plot(bit_lengths, mAP_at_100_list, marker='s', linestyle='--', color='#E74C3C', linewidth=2, markersize=8, label='mAP@100')
    plt.xlabel('Number of Bits', fontsize=12)
    plt.ylabel('Mean Average Precision', fontsize=12)
    plt.title('mAP and mAP@100 vs. Number of Bits', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'mAP_vs_bits.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create Storage and Query Time vs Bits plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot storage on primary y-axis
    ax1.plot(bit_lengths, [s/1024 for s in storage_list], marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Bits', fontsize=12)
    ax1.set_ylabel('Storage (KB)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis and plot query time
    ax2 = ax1.twinx()
    ax2.plot(bit_lengths, [t * 1000 for t in query_time_list], marker='o', linestyle='-', color='red', linewidth=2, markersize=8)
    ax2.set_ylabel('Query Time (ms)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Storage Consumption and Query Time vs. Number of Bits', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'storage_and_time_vs_bits.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create Precision-Recall curve
    plt.figure(figsize=(10, 6))
    for i, n_bits in enumerate(bit_lengths):
        plt.plot(recall_list[n_bits], precision_list[n_bits], 
                label=f'{n_bits}-bit', 
                linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'PR_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create recall curve
    plt.figure(figsize=(10, 6))
    for i, n_bits in enumerate(bit_lengths):
        plt.plot(recall_list[n_bits], label=f'{n_bits}-bit', linewidth=2)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall Curve', fontsize=14, pad=15)
    # log scale
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create precision curve
    plt.figure(figsize=(10, 6))
    for i, n_bits in enumerate(bit_lengths):
        plt.plot(precision_list[n_bits], label=f'{n_bits}-bit', linewidth=2)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision Curve', fontsize=14, pad=15)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'precision_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # save the mAP, storage, and query time in one file
    list_of_dicts = [
        {'mAP': mAP_list, 'mAP@100': mAP_at_100_list, 'storage': storage_list, 'query_time': query_time_list, 'bit_lengths': bit_lengths}
    ]
    np.save(os.path.join('outputs', 'metrics.npy'), list_of_dicts)


if __name__ == "__main__":
    main()
