"""
Synthetic data generation utilities for MLP demonstrations.
This module provides functions to create various 2D datasets suitable for
binary classification tasks.
"""

import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_classification

def generate_moons(n_samples=200, noise=0.2, random_state=42):
    """
    Generate a two-moons dataset for binary classification.
    
    This creates two interleaving half-moon shapes, which is a classic
    non-linearly separable dataset that demonstrates the power of neural networks.
    
    Args:
        n_samples (int): Number of data points to generate
        noise (float): Standard deviation of Gaussian noise added to data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is features array (n_samples, 2) and y is labels (n_samples,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)

def generate_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42):
    """
    Generate isotropic Gaussian blobs for clustering/classification.
    
    This creates clearly separable clusters, useful for demonstrating
    how neural networks can learn simple decision boundaries.
    
    Args:
        n_samples (int): Number of data points to generate
        centers (int): Number of cluster centers
        cluster_std (float): Standard deviation of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is features array (n_samples, 2) and y is labels (n_samples,)
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
                      cluster_std=cluster_std, random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)

def generate_classification(n_samples=200, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42):
    """
    Generate a random n-class classification problem.
    
    This creates a more general classification dataset with configurable complexity.
    
    Args:
        n_samples (int): Number of data points to generate
        n_features (int): Total number of features
        n_redundant (int): Number of redundant features
        n_informative (int): Number of informative features
        n_clusters_per_class (int): Number of clusters per class
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is features array (n_samples, n_features) and y is labels (n_samples,)
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                             n_redundant=n_redundant, n_informative=n_informative,
                             n_clusters_per_class=n_clusters_per_class,
                             random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)

def add_noise_to_data(X, noise_level=0.1):
    """
    Add Gaussian noise to existing data.
    
    Args:
        X (np.ndarray): Input data
        noise_level (float): Standard deviation of noise to add
        
    Returns:
        np.ndarray: Noisy version of input data
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise.astype(np.float32)