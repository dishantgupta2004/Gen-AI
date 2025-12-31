"""
Page 2: Feedforward Computation

This page demonstrates how data flows through a neural network layer by layer,
showing the mathematical operations at each step.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

def main():
    """Main function for the feedforward computation page."""
    st.title("➡️ Feedforward Computation")
    
    st.markdown("""
    **Feedforward** is the process of passing input data through the neural network 
    to generate an output. Let's see exactly how this works step by step!
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Network Configuration")
    
    # Simple network architecture for demonstration
    input_size = st.sidebar.slider("Input Size", 2, 5, 2)
    hidden_size = st.sidebar.slider("Hidden Layer Size", 2, 8, 3)
    output_size = st.sidebar.slider("Output Size", 1, 5, 2)
    
    activation_func = st.sidebar.selectbox(
        "Activation Function",
        ["ReLU", "tanh", "sigmoid"]
    )
    
    # Create a simple network for demonstration
    st.subheader("🏗️ Network Architecture")
    st.write(f"Input Layer: {input_size} neurons")
    st.write(f"Hidden Layer: {hidden_size} neurons ({activation_func})")
    st.write(f"Output Layer: {output_size} neurons")
    
    # Manual input values
    st.subheader("📝 Input Values")
    input_values = []
    cols = st.columns(input_size)
    for i in range(input_size):
        with cols[i]:
            val = st.number_input(f"Input {i+1}", value=1.0, step=0.1, key=f"input_{i}")
            input_values.append(val)
    
    input_array = np.array(input_values, dtype=np.float32)
    
    # Create a simple MLP for demonstration
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, activation):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            
            if activation == "ReLU":
                self.activation = nn.ReLU()
            elif activation == "tanh":
                self.activation = nn.Tanh()
            else:  # sigmoid
                self.activation = nn.Sigmoid()
        
        def forward_with_intermediates(self, x):
            # First layer computation
            z1 = self.fc1(x)  # Linear transformation
            a1 = self.activation(z1)  # Activation
            
            # Output layer
            z2 = self.fc2(a1)  # Linear transformation (no activation for output)
            
            return {
                'input': x,
                'hidden_linear': z1,
                'hidden_activated': a1,
                'output': z2
            }
    
    # Initialize model and get intermediate values
    model = SimpleMLP(input_size, hidden_size, output_size, activation_func)
    
    # Convert input to tensor
    input_tensor = torch.FloatTensor(input_array).unsqueeze(0)  # Add batch dimension
    
    # Get forward pass results
    with torch.no_grad():
        results = model.forward_with_intermediates(input_tensor)
    
    # Display step-by-step computation
    st.subheader("🔄 Step-by-Step Computation")
    
    # Step 1: Input to Hidden Layer
    st.write("**Step 1: Input → Hidden Layer**")
    
    # Get weights and biases
    W1 = model.fc1.weight.data.numpy()
    b1 = model.fc1.bias.data.numpy()
    
    st.write("Linear Transformation: z₁ = W₁ × input + b₁")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weights (W₁):**")
        st.dataframe(pd.DataFrame(W1, columns=[f"Input {i+1}" for i in range(input_size)]))
        
        st.write("**Biases (b₁):**")
        st.write(b1)
    
    with col2:
        st.write("**Computation:**")
        z1_values = results['hidden_linear'].squeeze().numpy()
        
        # Show the computation for each hidden neuron
        for i in range(hidden_size):
            computation = " + ".join([f"{W1[i,j]:.3f}×{input_values[j]:.3f}" for j in range(input_size)])
            computation += f" + {b1[i]:.3f}"
            st.write(f"z₁[{i+1}] = {computation} = {z1_values[i]:.3f}")
    
    # Activation step
    st.write(f"**Activation Function ({activation_func}):**")
    a1_values = results['hidden_activated'].squeeze().numpy()
    
    activation_df = pd.DataFrame({
        'Before Activation (z₁)': z1_values,
        f'After Activation (a₁)': a1_values
    })
    st.dataframe(activation_df)
    
    # Step 2: Hidden to Output Layer
    st.write("**Step 2: Hidden Layer → Output**")
    
    W2 = model.fc2.weight.data.numpy()
    b2 = model.fc2.bias.data.numpy()
    
    st.write("Linear Transformation: output = W₂ × a₁ + b₂")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weights (W₂):**")
        st.dataframe(pd.DataFrame(W2, columns=[f"Hidden {i+1}" for i in range(hidden_size)]))
        
        st.write("**Biases (b₂):**")
        st.write(b2)
    
    with col2:
        st.write("**Final Output:**")
        output_values = results['output'].squeeze().numpy()
        
        for i in range(output_size):
            computation = " + ".join([f"{W2[i,j]:.3f}×{a1_values[j]:.3f}" for j in range(hidden_size)])
            computation += f" + {b2[i]:.3f}"
            st.write(f"output[{i+1}] = {computation} = {output_values[i]:.3f}")
    
    # Visualization of the computation flow
    st.subheader("📊 Computation Flow Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layer positions
    input_positions = [(0, i) for i in range(input_size)]
    hidden_positions = [(2, i) for i in range(hidden_size)]
    output_positions = [(4, i) for i in range(output_size)]
    
    # Draw neurons
    for i, (x, y) in enumerate(input_positions):
        circle = plt.Circle((x, y), 0.1, color='lightblue')
        ax.add_patch(circle)
        ax.text(x, y, f'{input_values[i]:.2f}', ha='center', va='center', fontsize=8)
        ax.text(x-0.3, y, f'x{i+1}', ha='center', va='center', fontsize=10)
    
    for i, (x, y) in enumerate(hidden_positions):
        circle = plt.Circle((x, y), 0.1, color='lightgreen')
        ax.add_patch(circle)
        ax.text(x, y, f'{a1_values[i]:.2f}', ha='center', va='center', fontsize=8)
        ax.text(x-0.3, y, f'h{i+1}', ha='center', va='center', fontsize=10)
    
    for i, (x, y) in enumerate(output_positions):
        circle = plt.Circle((x, y), 0.1, color='lightcoral')
        ax.add_patch(circle)
        ax.text(x, y, f'{output_values[i]:.2f}', ha='center', va='center', fontsize=8)
        ax.text(x+0.3, y, f'y{i+1}', ha='center', va='center', fontsize=10)
    
    # Draw connections (simplified - just a few key ones)
    for i, (x1, y1) in enumerate(input_positions):
        for j, (x2, y2) in enumerate(hidden_positions):
            ax.plot([x1+0.1, x2-0.1], [y1, y2], 'k-', alpha=0.3, linewidth=1)
    
    for i, (x1, y1) in enumerate(hidden_positions):
        for j, (x2, y2) in enumerate(output_positions):
            ax.plot([x1+0.1, x2-0.1], [y1, y2], 'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, max(input_size, hidden_size, output_size))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Computation Flow', fontsize=14, fontweight='bold')
    
    # Add layer labels
    ax.text(0, -0.3, 'Input Layer', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, -0.3, 'Hidden Layer', ha='center', fontsize=12, fontweight='bold')
    ax.text(4, -0.3, 'Output Layer', ha='center', fontsize=12, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()
    
    # Educational content
    st.subheader("🎓 Key Concepts")
    
    with st.expander("Mathematical Formula", expanded=False):
        st.latex(r'''
        \text{For a layer with input } \mathbf{x} \text{, weight matrix } \mathbf{W} \text{, and bias } \mathbf{b}:
        ''')
        st.latex(r'''
        \mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}
        ''')
        st.latex(r'''
        \mathbf{a} = \sigma(\mathbf{z})
        ''')
        st.write("where σ is the activation function")
    
    with st.expander("Why Activation Functions?", expanded=False):
        st.markdown("""
        Without activation functions, multiple layers would be equivalent to a single layer!
        The activation function introduces **non-linearity**, allowing the network to learn
        complex patterns and decision boundaries.
        
        - **Linear combination**: z = Wx + b (just matrix multiplication)
        - **Non-linear activation**: a = σ(z) (introduces curves and complexity)
        """)
    
    with st.expander("Matrix Multiplication Details", expanded=False):
        st.markdown("""
        Each layer performs a **linear transformation** followed by an **activation function**:
        
        1. **Linear Transformation**: Combines all inputs with learned weights
        2. **Bias Addition**: Adds a learnable offset to each neuron  
        3. **Activation**: Applies non-linear function element-wise
        
        This process repeats for each layer, gradually transforming the input
        into the desired output representation.
        """)

if __name__ == "__main__":
    main()