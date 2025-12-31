"""
Page 4: Role of Weights and Biases

This page demonstrates how weights and biases affect neural network behavior
through interactive visualizations and simple examples.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def single_neuron_output(x1, x2, w1, w2, bias, activation='linear'):
    """Compute output of a single neuron."""
    z = w1 * x1 + w2 * x2 + bias
    
    if activation == 'ReLU':
        return max(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    else:  # linear
        return z

def plot_decision_line(w1, w2, bias, x_range=(-3, 3), y_range=(-3, 3)):
    """Plot the decision boundary for a single neuron (linear classifier)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute the neuron output for each point
    Z = w1 * X + w2 * Y + bias
    
    # Plot the decision boundary (where z = 0)
    ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=3)
    ax.contourf(X, Y, Z, levels=[-100, 0, 100], colors=['lightblue', 'lightcoral'], alpha=0.5)
    
    # Add labels and formatting
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('x₁ (Feature 1)', fontsize=12)
    ax.set_ylabel('x₂ (Feature 2)', fontsize=12)
    ax.set_title(f'Decision Boundary: {w1:.2f}x₁ + {w2:.2f}x₂ + {bias:.2f} = 0', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Positive Region\n(Output > 0)', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            verticalalignment='top', fontsize=10)
    ax.text(0.02, 0.02, 'Negative Region\n(Output < 0)', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            verticalalignment='bottom', fontsize=10)
    
    return fig

def main():
    """Main function for the weights and biases page."""
    st.title("⚖️ Role of Weights and Biases")
    
    st.markdown("""
    **Weights** and **biases** are the learnable parameters that determine how a neural network processes information.
    Let's explore how they affect the network's behavior!
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Single Neuron Parameters")
    
    # Weight controls
    st.sidebar.subheader("Weights")
    w1 = st.sidebar.slider("Weight 1 (w₁)", -5.0, 5.0, 1.0, 0.1, 
                          help="Controls influence of first input")
    w2 = st.sidebar.slider("Weight 2 (w₂)", -5.0, 5.0, 1.0, 0.1,
                          help="Controls influence of second input")
    
    # Bias control
    st.sidebar.subheader("Bias")
    bias = st.sidebar.slider("Bias (b)", -5.0, 5.0, 0.0, 0.1,
                            help="Shifts the decision boundary")
    
    # Activation function
    activation = st.sidebar.selectbox("Activation Function", 
                                     ["linear", "ReLU", "tanh", "sigmoid"])
    
    # Main content
    st.subheader("🧮 Single Neuron Computation")
    
    # Display the neuron equation
    st.latex(f"z = {w1:.2f} \\cdot x_1 + {w2:.2f} \\cdot x_2 + {bias:.2f}")
    
    if activation != "linear":
        st.latex(f"output = {activation}(z)")
    else:
        st.latex("output = z")
    
    # Interactive input testing
    st.subheader("🎯 Test Specific Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x1_test = st.number_input("Input x₁", value=1.0, step=0.1)
        x2_test = st.number_input("Input x₂", value=1.0, step=0.1)
    
    with col2:
        # Calculate output
        output = single_neuron_output(x1_test, x2_test, w1, w2, bias, activation)
        linear_output = w1 * x1_test + w2 * x2_test + bias
        
        st.metric("Linear Output (z)", f"{linear_output:.4f}")
        st.metric(f"Final Output ({activation})", f"{output:.4f}")
    
    # Decision boundary visualization (for 2D inputs)
    st.subheader("📊 Decision Boundary Visualization")
    
    fig = plot_decision_line(w1, w2, bias)
    
    # Add the test point to the plot
    ax = fig.gca()
    ax.plot(x1_test, x2_test, 'ko', markersize=10, label=f'Test Point ({x1_test:.1f}, {x2_test:.1f})')
    ax.legend()
    
    st.pyplot(fig)
    plt.close()
    
    # Weight magnitude analysis
    st.subheader("📏 Understanding Weight Magnitudes")
    
    # Show effect of different weight magnitudes
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Small Weights (Gentle Slope)**")
        fig_small = plot_decision_line(0.5, 0.5, 0)
        fig_small.suptitle("Small Weights: 0.5x₁ + 0.5x₂ = 0")
        st.pyplot(fig_small)
        plt.close()
    
    with col2:
        st.write("**Large Weights (Steep Slope)**")
        fig_large = plot_decision_line(3.0, 3.0, 0)
        fig_large.suptitle("Large Weights: 3.0x₁ + 3.0x₂ = 0")
        st.pyplot(fig_large)
        plt.close()
    
    # Bias effect demonstration
    st.subheader("📐 Understanding Bias Effects")
    
    st.write("Bias shifts the decision boundary without changing its orientation:")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    biases = [-2, 0, 2]
    titles = ["Negative Bias", "Zero Bias", "Positive Bias"]
    
    for i, (b, title) in enumerate(zip(biases, titles)):
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = 1.0 * X + 1.0 * Y + b
        
        axes[i].contour(X, Y, Z, levels=[0], colors='red', linewidths=3)
        axes[i].contourf(X, Y, Z, levels=[-100, 0, 100], colors=['lightblue', 'lightcoral'], alpha=0.5)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_xlabel('x₁')
        axes[i].set_ylabel('x₂')
        axes[i].set_title(f'{title}\n1.0x₁ + 1.0x₂ + {b} = 0')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Multi-layer network demonstration
    st.subheader("🏗️ Weights in Multi-Layer Networks")
    
    st.markdown("""
    In deeper networks, weights in different layers serve different purposes:
    - **Early layers**: Extract basic features and patterns
    - **Middle layers**: Combine features into more complex representations  
    - **Final layers**: Make the final decision based on learned representations
    """)
    
    # Create a simple 3-layer network visualization
    if st.button("Visualize Random Network Weights"):
        # Create a simple network
        torch.manual_seed(42)  # For reproducible results
        network = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        
        # Display weight matrices
        st.write("**Layer 1 Weights (Input → Hidden 1):**")
        w1 = network[0].weight.data.numpy()
        b1 = network[0].bias.data.numpy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Weights:")
            st.dataframe(w1)
        with col2:
            st.write("Biases:")
            st.write(b1)
        
        st.write("**Layer 2 Weights (Hidden 1 → Hidden 2):**")
        w2 = network[2].weight.data.numpy()
        b2 = network[2].bias.data.numpy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Weights:")
            st.dataframe(w2)
        with col2:
            st.write("Biases:")
            st.write(b2)
        
        st.write("**Layer 3 Weights (Hidden 2 → Output):**")
        w3 = network[4].weight.data.numpy()
        b3 = network[4].bias.data.numpy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Weights:")
            st.dataframe(w3)
        with col2:
            st.write("Biases:")
            st.write(b3)
        
        # Show weight distributions
        st.write("**Weight Distributions Across Layers:**")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        weights_list = [w1.flatten(), w2.flatten(), w3.flatten()]
        layer_names = ["Layer 1", "Layer 2", "Layer 3"]
        
        for i, (weights, name) in enumerate(zip(weights_list, layer_names)):
            axes[i].hist(weights, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{name} Weight Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Weight initialization discussion
    st.subheader("🎲 Weight Initialization")
    
    initialization_method = st.selectbox(
        "Select initialization method to explore:",
        ["Xavier/Glorot", "He/Kaiming", "Random Normal", "All Zeros", "All Ones"]
    )
    
    # Generate weights based on selected method
    input_size, output_size = 100, 50
    
    if initialization_method == "Xavier/Glorot":
        weights = np.random.normal(0, np.sqrt(2.0 / (input_size + output_size)), (output_size, input_size))
        description = "Good for tanh and sigmoid activations. Keeps variance stable across layers."
    elif initialization_method == "He/Kaiming":
        weights = np.random.normal(0, np.sqrt(2.0 / input_size), (output_size, input_size))
        description = "Designed for ReLU activations. Prevents vanishing gradients in deep networks."
    elif initialization_method == "Random Normal":
        weights = np.random.normal(0, 1, (output_size, input_size))
        description = "Simple random initialization. May cause vanishing/exploding gradients."
    elif initialization_method == "All Zeros":
        weights = np.zeros((output_size, input_size))
        description = "Bad initialization! All neurons learn the same thing (symmetry problem)."
    else:  # All Ones
        weights = np.ones((output_size, input_size))
        description = "Also bad! Neurons don't learn different features."
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{initialization_method} Initialization**")
        st.write(description)
        
        # Plot weight distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{initialization_method} Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Statistics:**")
        st.metric("Mean", f"{np.mean(weights):.6f}")
        st.metric("Standard Deviation", f"{np.std(weights):.6f}")
        st.metric("Min Value", f"{np.min(weights):.6f}")
        st.metric("Max Value", f"{np.max(weights):.6f}")
    
    # Educational content
    st.subheader("🎓 Key Concepts")
    
    with st.expander("What do weights represent?", expanded=False):
        st.markdown("""
        **Weights** determine the strength and direction of connections between neurons:
        
        - **Positive weights**: Excitatory connections (increase activation)
        - **Negative weights**: Inhibitory connections (decrease activation)
        - **Large magnitude**: Strong influence on the next layer
        - **Small magnitude**: Weak influence on the next layer
        - **Zero weight**: No connection (neuron is ignored)
        """)
    
    with st.expander("What do biases do?", expanded=False):
        st.markdown("""
        **Biases** allow neurons to activate even when all inputs are zero:
        
        - **Positive bias**: Makes the neuron more likely to activate
        - **Negative bias**: Makes the neuron less likely to activate
        - **Bias = 0**: Neuron only activates based on weighted inputs
        
        Think of bias as the "default behavior" of a neuron before considering inputs.
        """)
    
    with st.expander("Why is initialization important?", expanded=False):
        st.markdown("""
        Proper weight initialization is crucial for training success:
        
        **Good initialization**:
        - Prevents vanishing/exploding gradients
        - Ensures neurons learn different features
        - Speeds up convergence
        
        **Bad initialization**:
        - All zeros → All neurons learn the same thing
        - Too large → Exploding gradients
        - Too small → Vanishing gradients
        """)
    
    with st.expander("How do networks learn weights?", expanded=False):
        st.markdown("""
        Networks learn optimal weights through **gradient descent**:
        
        1. **Forward pass**: Compute predictions using current weights
        2. **Loss calculation**: Compare predictions to true values
        3. **Backward pass**: Calculate gradients (how to change weights)
        4. **Weight update**: Adjust weights in the direction that reduces loss
        5. **Repeat**: Continue until weights converge to good values
        
        The learning rate controls how big steps we take when updating weights.
        """)

if __name__ == "__main__":
    main()