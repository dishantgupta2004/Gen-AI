"""
Training utilities for MLP models.
This module provides functions for training PyTorch models and tracking performance.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(model, X_train, y_train, lr=0.01, epochs=100, verbose=False):
    """
    Train a PyTorch model on the given data using stochastic gradient descent.
    
    Args:
        model (torch.nn.Module): The neural network model to train
        X_train (np.ndarray): Training features of shape (n_samples, n_features)
        y_train (np.ndarray): Training labels of shape (n_samples,)
        lr (float): Learning rate for the optimizer
        epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
        
    Returns:
        list: List of loss values for each epoch
    """
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)
    
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Zero the gradients (PyTorch accumulates gradients by default)
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(X_tensor)
        
        # Compute loss
        loss = criterion(outputs, y_tensor)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Store loss for plotting
        losses.append(loss.item())
        
        # Print progress occasionally
        if verbose and (epoch + 1) % (epochs // 10) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

def train_with_validation(model, X, y, lr=0.01, epochs=100, validation_split=0.2, random_state=42):
    """
    Train a model with validation set for monitoring overfitting.
    
    Args:
        model (torch.nn.Module): The neural network model to train
        X (np.ndarray): All features
        y (np.ndarray): All labels
        lr (float): Learning rate
        epochs (int): Number of training epochs
        validation_split (float): Fraction of data to use for validation
        random_state (int): Random seed for train/validation split
        
    Returns:
        dict: Dictionary containing 'train_losses', 'val_losses', 'train_accuracies', 'val_accuracies'
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
            # Calculate accuracies
            train_preds = torch.argmax(train_outputs, dim=1).numpy()
            val_preds = torch.argmax(val_outputs, dim=1).numpy()
            
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
        
        # Store metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (torch.nn.Module): Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        dict: Dictionary with accuracy and loss metrics
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        y_tensor = torch.from_numpy(y_test)
        
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
        
        accuracy = accuracy_score(y_test, predictions)
        loss = nn.CrossEntropyLoss()(outputs, y_tensor).item()
        
    return {'accuracy': accuracy, 'loss': loss}