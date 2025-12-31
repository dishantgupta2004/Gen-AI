"""
Plotting utilities for visualizing MLP behavior and training progress.
This module provides functions for creating decision boundary plots, loss curves,
and other visualizations to help understand neural network behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import torch

def plot_decision_boundary(model, X, y, title="Decision Boundary", resolution=100):
    """
    Plot the decision boundary of a trained model over 2D data.
    
    This function creates a meshgrid over the input space and uses the model
    to predict the class for each point, then visualizes the decision regions.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        X (np.ndarray): Input features of shape (n_samples, 2)
        y (np.ndarray): True labels of shape (n_samples,)
        title (str): Title for the plot
        resolution (int): Resolution of the meshgrid (higher = smoother boundaries)
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    # Create a meshgrid covering the data range with some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Flatten the meshgrid and feed into model for predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get model predictions for the grid
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid_points.astype(np.float32))
        predictions = model(grid_tensor)
        Z = torch.argmax(predictions, dim=1).numpy().reshape(xx.shape)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision regions with low alpha for background
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    
    # Plot the actual data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    
    # Add colorbar for the decision regions
    plt.colorbar(contour, ax=ax, label='Predicted Class')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_loss_curve(losses, title="Training Loss", validation_losses=None):
    """
    Plot the training loss curve over epochs.
    
    Args:
        losses (list): List of training loss values
        title (str): Title for the plot
        validation_losses (list, optional): List of validation loss values
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    
    if validation_losses is not None:
        ax.plot(epochs, validation_losses, 'r-', linewidth=2, label='Validation Loss')
        ax.legend()
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_activation_functions():
    """
    Plot and compare different activation functions.
    
    Returns:
        matplotlib.figure.Figure: The generated comparison plot
    """
    x = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    relu = np.maximum(0, x)
    tanh = np.tanh(x)  
    sigmoid = 1 / (1 + np.exp(-x))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ReLU
    axes[0].plot(x, relu, 'b-', linewidth=2)
    axes[0].set_title('ReLU Activation', fontweight='bold')
    axes[0].set_xlabel('Input')
    axes[0].set_ylabel('Output')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tanh
    axes[1].plot(x, tanh, 'g-', linewidth=2)
    axes[1].set_title('Tanh Activation', fontweight='bold')
    axes[1].set_xlabel('Input')
    axes[1].set_ylabel('Output')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Sigmoid
    axes[2].plot(x, sigmoid, 'r-', linewidth=2)
    axes[2].set_title('Sigmoid Activation', fontweight='bold')
    axes[2].set_xlabel('Input')
    axes[2].set_ylabel('Output')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics_dict):
    """
    Plot training and validation metrics side by side.
    
    Args:
        metrics_dict (dict): Dictionary containing training metrics
        
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(metrics_dict['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, metrics_dict['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, metrics_dict['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, metrics_dict['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, metrics_dict['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_network_diagram(layer_sizes):
    """
    Create a simple text-based representation of the network architecture.
    
    Args:
        layer_sizes (list): List of layer sizes
        
    Returns:
        str: ASCII art representation of the network
    """
    diagram = "Network Architecture:\n"
    diagram += "=" * 30 + "\n"
    
    for i, size in enumerate(layer_sizes):
        if i == 0:
            layer_type = "Input Layer"
        elif i == len(layer_sizes) - 1:
            layer_type = "Output Layer"
        else:
            layer_type = f"Hidden Layer {i}"
        
        # Create visual representation of neurons
        neurons_visual = "○ " * min(size, 8)  # Limit visual neurons to 8
        if size > 8:
            neurons_visual += f"... ({size} total)"
        
        diagram += f"{layer_type:>15}: {neurons_visual}\n"
        
        # Add arrows between layers
        if i < len(layer_sizes) - 1:
            diagram += " " * 17 + "↓\n"
    
    return diagram