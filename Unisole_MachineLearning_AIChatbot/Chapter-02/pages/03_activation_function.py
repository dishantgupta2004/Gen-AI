"""
Page 3: Activation Functions

This page provides an interactive comparison of different activation functions
and shows how they affect neural network behavior.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.plotting import plot_activation_functions
from utils.data import generate_moons
from models.mlp import MLP
from utils.training import train_model
from utils.plotting import plot_decision_boundary

def compute_activation(x, activation_type):
    """Compute activation function values."""
    if activation_type == "ReLU":
        return np.maximum(0, x)
    elif activation_type == "tanh":
        return np.tanh(x)
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    elif activation_type == "Leaky ReLU":
        return np.where(x > 0, x, 0.01 * x)
    elif activation_type == "ELU":
        return np.where(x > 0, x, np.exp(x) - 1)
    else:
        return x  # Linear

def compute_derivative(x, activation_type):
    """Compute derivative of activation function."""
    if activation_type == "ReLU":
        return np.where(x > 0, 1, 0)
    elif activation_type == "tanh":
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2
    elif activation_type == "sigmoid":
        sig_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sig_x * (1 - sig_x)
    elif activation_type == "Leaky ReLU":
        return np.where(x > 0, 1, 0.01)
    elif activation_type == "ELU":
        return np.where(x > 0, 1, np.exp(x))
    else:
        return np.ones_like(x)  # Linear

def main():
    """Main function for the activation functions page."""
    st.title("⚡ Activation Functions")
    
    st.markdown("""
    **Activation functions** are mathematical functions that determine the output of neural network nodes.
    They introduce non-linearity into the network, enabling it to learn complex patterns.
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Function Comparison")
    
    # Select activation functions to compare
    available_functions = ["ReLU", "tanh", "sigmoid", "Leaky ReLU", "ELU", "Linear"]
    selected_functions = st.sidebar.multiselect(
        "Select Functions to Compare",
        available_functions,
        default=["ReLU", "tanh", "sigmoid"]
    )
    
    if not selected_functions:
        st.warning("Please select at least one activation function to compare.")
        return
    
    # Input range for plotting
    x_min = st.sidebar.slider("Input Range (Min)", -10.0, 0.0, -5.0)
    x_max = st.sidebar.slider("Input Range (Max)", 0.0, 10.0, 5.0)
    
    # Generate input values
    x = np.linspace(x_min, x_max, 1000)
    
    # Plot comparison
    st.subheader("📊 Function Comparison")
    
    # Create two columns for function and derivative plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Activation Functions**")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, func in enumerate(selected_functions):
            y = compute_activation(x, func)
            ax1.plot(x, y, label=func, color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel("Input (x)")
        ax1.set_ylabel("Output f(x)")
        ax1.set_title("Activation Functions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.write("**Derivatives (Gradients)**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        for i, func in enumerate(selected_functions):
            y_prime = compute_derivative(x, func)
            ax2.plot(x, y_prime, label=f"{func} derivative", color=colors[i % len(colors)], linewidth=2)
        
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Derivative f'(x)")
        ax2.set_title("Activation Function Derivatives")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        st.pyplot(fig2)
        plt.close()
    
    # Interactive single function explorer
    st.subheader("🔍 Interactive Function Explorer")
    
    selected_function = st.selectbox(
        "Choose a function to explore in detail:",
        available_functions
    )
    
    # Input value slider
    input_value = st.slider(
        "Input Value",
        float(x_min), float(x_max), 0.0, 0.1
    )
    
    # Calculate output and derivative at the selected point
    output_value = compute_activation(np.array([input_value]), selected_function)[0]
    derivative_value = compute_derivative(np.array([input_value]), selected_function)[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input", f"{input_value:.2f}")
    with col2:
        st.metric(f"{selected_function} Output", f"{output_value:.4f}")
    with col3:
        st.metric("Derivative", f"{derivative_value:.4f}")
    
    # Plot the selected function with the point highlighted
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Function plot
    y_selected = compute_activation(x, selected_function)
    ax3.plot(x, y_selected, 'b-', linewidth=2, label=selected_function)
    ax3.plot(input_value, output_value, 'ro', markersize=10, label=f'Point ({input_value:.2f}, {output_value:.4f})')
    ax3.set_xlabel("Input (x)")
    ax3.set_ylabel("Output f(x)")
    ax3.set_title(f"{selected_function} Activation Function")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Derivative plot
    y_prime_selected = compute_derivative(x, selected_function)
    ax4.plot(x, y_prime_selected, 'g-', linewidth=2, label=f"{selected_function} derivative")
    ax4.plot(input_value, derivative_value, 'ro', markersize=10, label=f'Gradient: {derivative_value:.4f}')
    ax4.set_xlabel("Input (x)")
    ax4.set_ylabel("Derivative f'(x)")
    ax4.set_title(f"{selected_function} Derivative")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    st.pyplot(fig3)
    plt.close()
    
    # Practical comparison with neural networks
    st.subheader("🧪 Impact on Neural Network Learning")
    
    st.markdown("""
    Let's see how different activation functions affect a neural network's ability to learn!
    We'll train identical networks with different activation functions on the same dataset.
    """)
    
    # Generate a dataset
    X, y = generate_moons(n_samples=300, noise=0.3)
    
    if st.button("Compare Network Performance", type="primary"):
        comparison_functions = ["ReLU", "tanh", "sigmoid"]
        
        fig4, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, activation in enumerate(comparison_functions):
            with st.spinner(f"Training network with {activation} activation..."):
                # Create and train model
                model = MLP([2, 8, 8, 2], activation=activation)
                losses = train_model(model, X, y, lr=0.01, epochs=200)
                
                # Plot decision boundary
                axes[i] = plt.subplot(1, 3, i+1)
                
                # Create meshgrid for decision boundary
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                   np.linspace(y_min, y_max, 100))
                
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                model.eval()
                with torch.no_grad():
                    Z = model(torch.from_numpy(grid_points.astype(np.float32)))
                    Z = torch.argmax(Z, dim=1).numpy().reshape(xx.shape)
                
                axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
                axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='black')
                axes[i].set_title(f'{activation} Activation\nFinal Loss: {losses[-1]:.4f}')
                axes[i].set_xlabel('Feature 1')
                axes[i].set_ylabel('Feature 2')
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    # Educational content about each activation function
    st.subheader("📚 Activation Function Properties")
    
    function_info = {
        "ReLU": {
            "formula": "f(x) = max(0, x)",
            "range": "[0, ∞)",
            "pros": ["Fast computation", "Helps with vanishing gradient", "Sparse activation"],
            "cons": ["Dead neurons problem", "Not zero-centered", "Unbounded output"]
        },
        "tanh": {
            "formula": "f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)",
            "range": "(-1, 1)",
            "pros": ["Zero-centered output", "Smooth gradient", "Bounded output"],
            "cons": ["Vanishing gradient problem", "Slower than ReLU"]
        },
        "sigmoid": {
            "formula": "f(x) = 1 / (1 + e⁻ˣ)",
            "range": "(0, 1)",
            "pros": ["Smooth gradient", "Output interpretable as probability", "Bounded output"],
            "cons": ["Vanishing gradient problem", "Not zero-centered", "Slower computation"]
        }
    }
    
    selected_info = st.selectbox("Select function for detailed information:", list(function_info.keys()))
    
    if selected_info in function_info:
        info = function_info[selected_info]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Formula:** {info['formula']}")
            st.write(f"**Range:** {info['range']}")
        
        with col2:
            st.write("**Advantages:**")
            for pro in info['pros']:
                st.write(f"• {pro}")
            
            st.write("**Disadvantages:**")
            for con in info['cons']:
                st.write(f"• {con}")
    
    # Key takeaways
    with st.expander("🎯 Key Takeaways", expanded=False):
        st.markdown("""
        1. **ReLU** is the most commonly used activation function due to its simplicity and effectiveness
        2. **tanh** works better than sigmoid for hidden layers because it's zero-centered
        3. **Sigmoid** is often used in the output layer for binary classification
        4. The **derivative** of the activation function affects how fast the network learns
        5. **Vanishing gradients** occur when derivatives become very small (common with sigmoid/tanh)
        6. **Dead neurons** can occur with ReLU when they always output zero
        """)

if __name__ == "__main__":
    main()