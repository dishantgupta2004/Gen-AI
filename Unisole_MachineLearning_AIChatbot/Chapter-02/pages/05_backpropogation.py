"""
Page 5: Gradient Descent and Backpropagation

This page explains how neural networks learn through backpropagation and gradient descent,
with visualizations and interactive examples.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.data import generate_moons
from models.mlp import MLP
from utils.plotting import plot_loss_curve

def simple_gradient_example():
    """Demonstrate gradient descent on a simple quadratic function."""
    # Define a simple quadratic function: f(x) = (x - 2)^2 + 1
    def f(x):
        return (x - 2)**2 + 1
    
    def df_dx(x):
        return 2 * (x - 2)
    
    return f, df_dx

def visualize_gradient_descent(f, df_dx, start_x, learning_rate, num_steps):
    """Visualize gradient descent steps on a 1D function."""
    x_vals = [start_x]
    y_vals = [f(start_x)]
    
    current_x = start_x
    
    for _ in range(num_steps):
        gradient = df_dx(current_x)
        current_x = current_x - learning_rate * gradient
        x_vals.append(current_x)
        y_vals.append(f(current_x))
    
    return x_vals, y_vals

def main():
    """Main function for the backpropagation page."""
    st.title("🔄 Gradient Descent and Backpropagation")
    
    st.markdown("""
    **Backpropagation** is the algorithm that allows neural networks to learn from their mistakes.
    It uses **gradient descent** to adjust weights and biases to minimize the prediction error.
    """)
    
    # Section 1: Gradient Descent Visualization
    st.subheader("📈 Understanding Gradient Descent")
    
    st.markdown("""
    Gradient descent is an optimization algorithm that finds the minimum of a function by
    following the negative gradient (steepest descent direction).
    """)
    
    # Sidebar controls for gradient descent demo
    st.sidebar.header("🎛️ Gradient Descent Demo")
    start_x = st.sidebar.slider("Starting Point", -2.0, 6.0, 5.0, 0.1)
    learning_rate_gd = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    num_steps = st.sidebar.slider("Number of Steps", 1, 50, 20)
    
    # Create the simple function and its derivative
    f, df_dx = simple_gradient_example()
    
    # Visualize gradient descent
    x_path, y_path = visualize_gradient_descent(f, df_dx, start_x, learning_rate_gd, num_steps)
    
    # Plot the function and gradient descent path
    x_range = np.linspace(-2, 6, 300)
    y_range = f(x_range)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Function plot with gradient descent path
    ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = (x-2)² + 1')
    ax1.plot(x_path, y_path, 'ro-', linewidth=2, markersize=6, alpha=0.7, label='Gradient Descent Path')
    ax1.plot(x_path[0], y_path[0], 'go', markersize=10, label='Start')
    ax1.plot(x_path[-1], y_path[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gradient Descent on f(x) = (x-2)² + 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convergence plot
    ax2.plot(range(len(y_path)), y_path, 'b-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence Over Steps')
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # Display final results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Starting Value", f"{f(start_x):.4f}")
    with col2:
        st.metric("Final Value", f"{y_path[-1]:.4f}")
    with col3:
        st.metric("Improvement", f"{f(start_x) - y_path[-1]:.4f}")
    
    # Section 2: Neural Network Training Visualization
    st.subheader("🧠 Neural Network Training Process")
    
    st.markdown("""
    Now let's see how gradient descent works in neural networks. We'll train a network
    step by step and visualize how the loss decreases and decision boundary changes.
    """)
    
    # Training parameters
    st.sidebar.header("🎛️ Training Parameters")
    epochs_demo = st.sidebar.slider("Training Epochs", 10, 200, 50)
    learning_rate_nn = st.sidebar.slider("Learning Rate (NN)", 0.001, 0.1, 0.01, 0.001)
    
    # Generate data
    X, y = generate_moons(n_samples=200, noise=0.2)
    
    if st.button("Train Network Step by Step", type="primary"):
        # Create model
        model = MLP([2, 6, 2], activation='ReLU')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_nn)
        
        # Convert data to tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        # Store training history
        losses = []
        weights_history = []
        
        # Training loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs_demo):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store history
            losses.append(loss.item())
            
            # Store weights for first layer (for visualization)
            if epoch % 10 == 0:
                weights_history.append(model.network[0].weight.data.clone().numpy())
            
            # Update progress
            progress = (epoch + 1) / epochs_demo
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{epochs_demo}, Loss: {loss.item():.4f}')
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Loss Curve**")
            fig_loss = plot_loss_curve(losses, "Training Loss")
            st.pyplot(fig_loss)
            plt.close()
        
        with col2:
            st.write("**Final Decision Boundary**")
            
            # Plot decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            model.eval()
            with torch.no_grad():
                Z = model(torch.from_numpy(grid_points.astype(np.float32)))
                Z = torch.argmax(Z, dim=1).numpy().reshape(xx.shape)
            
            fig_boundary, ax = plt.subplots(figsize=(8, 6))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='black')
            ax.set_title('Learned Decision Boundary')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            st.pyplot(fig_boundary)
            plt.close()
    
    # Section 3: Backpropagation Algorithm Explanation
    st.subheader("🔄 The Backpropagation Algorithm")
    
    st.markdown("""
    Backpropagation works by computing gradients layer by layer, starting from the output
    and working backwards through the network.
    """)
    
    # Interactive backprop explanation
    with st.expander("Step-by-Step Backpropagation", expanded=False):
        st.markdown("""
        **Forward Pass:**
        1. Input data flows through the network
        2. Each layer computes: output = activation(weights × input + bias)
        3. Final layer produces predictions
        4. Loss function compares predictions to true labels
        
        **Backward Pass:**
        1. Compute gradient of loss with respect to output layer
        2. Use chain rule to compute gradients for previous layers
        3. Continue backwards through all layers
        4. Update weights using computed gradients
        
        **Mathematical Foundation:**
        
        For a simple network: Input → Hidden → Output
        """)
        
        st.latex(r"""
        \text{Forward:} \quad h = \sigma(W_1 x + b_1), \quad y = W_2 h + b_2
        """)
        
        st.latex(r"""
        \text{Loss:} \quad L = \frac{1}{2}(y - t)^2
        """)
        
        st.latex(r"""
        \text{Backward:} \quad \frac{\partial L}{\partial W_2} = (y-t) \cdot h
        """)
        
        st.latex(r"""
        \frac{\partial L}{\partial W_1} = (y-t) \cdot W_2 \cdot \sigma'(W_1 x + b_1) \cdot x
        """)
    
    # Section 4: Learning Rate Effects
    st.subheader("⚡ Learning Rate Effects")
    
    st.markdown("""
    The learning rate controls how big steps we take during optimization.
    Let's see how different learning rates affect training:
    """)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    if st.button("Compare Learning Rates"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, lr in enumerate(learning_rates):
            with st.spinner(f"Training with learning rate {lr}..."):
                # Create fresh model for each learning rate
                model = MLP([2, 8, 2], activation='ReLU')
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                
                X_tensor = torch.from_numpy(X)
                y_tensor = torch.from_numpy(y)
                
                losses_lr = []
                for epoch in range(100):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    losses_lr.append(loss.item())
                
                # Plot loss curve
                axes[i].plot(losses_lr, linewidth=2)
                axes[i].set_title(f'Learning Rate: {lr}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True, alpha=0.3)
                
                # Add final loss as text
                final_loss = losses_lr[-1]
                axes[i].text(0.7, 0.9, f'Final Loss: {final_loss:.4f}', 
                           transform=axes[i].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Observations:**
        - **Too small** (0.001): Very slow convergence
        - **Good** (0.01): Steady, stable convergence  
        - **High** (0.1): Faster but potentially unstable
        - **Too high** (0.5): May oscillate or diverge
        """)
    
    # Section 5: Common Training Problems
    st.subheader("⚠️ Common Training Problems")
    
    problem_type = st.selectbox(
        "Select a training problem to learn about:",
        ["Vanishing Gradients", "Exploding Gradients", "Local Minima", "Overfitting"]
    )
    
    if problem_type == "Vanishing Gradients":
        st.markdown("""
        **Vanishing Gradients** occur when gradients become very small as they propagate backwards.
        
        **Causes:**
        - Deep networks with many layers
        - Activation functions like sigmoid/tanh
        - Poor weight initialization
        
        **Solutions:**
        - Use ReLU activations
        - Proper weight initialization (Xavier/He)
        - Batch normalization
        - Skip connections (ResNet)
        """)
        
    elif problem_type == "Exploding Gradients":
        st.markdown("""
        **Exploding Gradients** occur when gradients become very large during backpropagation.
        
        **Causes:**
        - Poor weight initialization
        - High learning rates
        - Deep networks without proper normalization
        
        **Solutions:**
        - Gradient clipping
        - Lower learning rates
        - Proper weight initialization
        - Batch normalization
        """)
        
    elif problem_type == "Local Minima":
        st.markdown("""
        **Local Minima** are points where the loss function has a local minimum but not global minimum.
        
        **Modern View:**
        - High-dimensional loss landscapes have many saddle points
        - Local minima are often "good enough" for practical purposes
        - SGD with momentum can escape shallow local minima
        
        **Solutions:**
        - Use momentum-based optimizers (Adam, RMSprop)
        - Random initialization with multiple training runs
        - Learning rate scheduling
        """)
        
    else:  # Overfitting
        st.markdown("""
        **Overfitting** occurs when the model memorizes training data but fails to generalize.
        
        **Signs:**
        - Training loss decreases but validation loss increases
        - Large gap between training and validation accuracy
        
        **Solutions:**
        - Regularization (L1/L2, Dropout)
        - Early stopping
        - More training data
        - Simpler model architecture
        """)
    
if __name__ == "__main__":
    main()