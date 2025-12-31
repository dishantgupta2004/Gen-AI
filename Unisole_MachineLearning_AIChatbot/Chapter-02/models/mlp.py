"""
PyTorch implementation of a configurable Multilayer Perceptron (MLP).
This module provides a flexible MLP class that can be customized with different
architectures and activation functions.
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A multilayer perceptron with configurable hidden layers and activation function.
    
    This class creates a feedforward neural network where you can specify:
    - The number of neurons in each layer
    - The activation function to use between layers
    - The network depth (number of hidden layers)
    
    Args:
        layer_sizes (list): List of integers specifying the number of neurons in each layer.
                           First element is input size, last is output size.
                           Example: [2, 4, 4, 2] creates a network with 2 inputs,
                           two hidden layers of 4 neurons each, and 2 outputs.
        activation (str): Activation function to use. Options: 'ReLU', 'tanh', 'sigmoid'
    """
    
    def __init__(self, layer_sizes, activation='ReLU'):
        super(MLP, self).__init__()
        
        # Validate inputs
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes")
        
        # Build a list of layers based on layer_sizes
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            # Add linear transformation (weights and biases)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation function after each hidden layer (not after output layer)
            if i < len(layer_sizes) - 2:  # if not the output layer
                if activation == 'ReLU':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        self.layer_sizes = layer_sizes
        self.activation = activation
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_features)
        """
        return self.network(x)
    
    def get_architecture_summary(self):
        """
        Return a string summary of the network architecture.
        
        Returns:
            str: Human-readable description of the network structure
        """
        summary = f"MLP Architecture:\n"
        summary += f"- Input Layer: {self.layer_sizes[0]} neurons\n"
        
        for i, size in enumerate(self.layer_sizes[1:-1], 1):
            summary += f"- Hidden Layer {i}: {size} neurons ({self.activation})\n"
        
        summary += f"- Output Layer: {self.layer_sizes[-1]} neurons\n"
        summary += f"- Total Parameters: {sum(p.numel() for p in self.parameters())}"
        
        return summary