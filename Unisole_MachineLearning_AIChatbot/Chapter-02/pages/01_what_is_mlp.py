"""
Page 1: What is a Multilayer Perceptron (MLP)?

This page provides an interactive introduction to MLPs using a binary classification task.
Users can adjust network architecture and training parameters to see how they affect
the model's ability to learn decision boundaries.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import our custom modules
from utils.data import generate_moons, generate_blobs, generate_classification
from models.mlp import MLP
from utils.training import train_model, evaluate_model
from utils.plotting import plot_decision_boundary, plot_loss_curve, create_network_diagram



def main():
    """Main function for the MLP introduction page."""
    
    st.session_state['current_page'] = '1_what_is_MLP'
    st.title("🧠 What is a Multilayer Perceptron (MLP)?")
    
    st.markdown("""
    A **Multilayer Perceptron (MLP)** is a type of artificial neural network that consists of:
    - An **input layer** that receives the data
    - One or more **hidden layers** that process the information
    - An **output layer** that produces the final prediction
    
    Each layer contains multiple **neurons** (nodes) that are connected to neurons in the next layer.
    The connections have **weights** that determine how much influence each neuron has on the next layer.
    """)
    
    # Create two columns: main content and chatbot
    col1, col2 = st.columns([2, 1])
    
    # Sidebar controls for model configuration
    st.sidebar.header("🎛️ Model Configuration")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Choose Dataset",
        ["Two Moons", "Gaussian Blobs", "Random Classification"],
        help="Different datasets to test the MLP's classification ability"
    )
    
    # Architecture parameters
    st.sidebar.subheader("Network Architecture")
    n_hidden_layers = st.sidebar.slider(
        "Number of Hidden Layers", 
        min_value=1, max_value=4, value=2,
        help="More layers can learn more complex patterns but may overfit"
    )
    
    neurons_per_layer = st.sidebar.slider(
        "Neurons per Hidden Layer", 
        min_value=2, max_value=32, value=8,
        help="More neurons increase model capacity but also complexity"
    )
    
    activation_function = st.sidebar.selectbox(
        "Activation Function",
        ["ReLU", "tanh", "sigmoid"],
        help="Function that determines neuron output given its input"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.001, max_value=0.5, value=0.01, step=0.001,
        help="How fast the model learns (too high = unstable, too low = slow)"
    )
    
    n_epochs = st.sidebar.slider(
        "Number of Epochs",
        min_value=10, max_value=500, value=100,
        help="How many times to go through the entire dataset"
    )
    
    # Dataset parameters
    st.sidebar.subheader("Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 500, 200)
    
    if dataset_type == "Two Moons":
        noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)
    elif dataset_type == "Gaussian Blobs":
        cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0, 0.1)
    
    # Generate the selected dataset
    if dataset_type == "Two Moons":
        X, y = generate_moons(n_samples=n_samples, noise=noise_level)
    elif dataset_type == "Gaussian Blobs":
        X, y = generate_blobs(n_samples=n_samples, cluster_std=cluster_std)
    else:  # Random Classification
        X, y = generate_classification(n_samples=n_samples)
    
    # Display dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training Data")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolor='black')
        ax.set_title(f"{dataset_type} Dataset", fontweight='bold', fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Class')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🏗️ Network Architecture")
        
        # Define the layer sizes: input (2) + hidden layers + output (2 for binary classification)
        layer_sizes = [2] + [neurons_per_layer] * n_hidden_layers + [2]
        
        # Display network diagram
        network_diagram = create_network_diagram(layer_sizes)
        st.text(network_diagram)
        
        # Create and display model summary
        model = MLP(layer_sizes, activation=activation_function)
        st.text(model.get_architecture_summary())
    
    # Training section
    st.subheader("🚀 Training the Model")
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training the neural network..."):
            
            # Initialize the model
            model = MLP(layer_sizes, activation=activation_function)
            
            # Train the model
            losses = train_model(model, X, y, lr=learning_rate, epochs=n_epochs)
            
            # Evaluate the model
            evaluation = evaluate_model(model, X, y)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Training Progress")
                fig_loss = plot_loss_curve(losses, "Training Loss Over Time")
                st.pyplot(fig_loss)
                plt.close()
                
                # Display final metrics
                st.metric("Final Training Accuracy", f"{evaluation['accuracy']:.3f}")
                st.metric("Final Training Loss", f"{evaluation['loss']:.4f}")
            
            with col2:
                st.subheader("🎯 Decision Boundary")
                fig_boundary = plot_decision_boundary(
                    model, X, y, 
                    title=f"Learned Decision Boundary\n({activation_function} activation)"
                )
                st.pyplot(fig_boundary)
                plt.close()
    
    # Educational content
    st.subheader("🎓 Key Concepts")
    
    with st.expander("How does an MLP work?", expanded=False):
        st.markdown("""
        1. **Input Layer**: Receives the input features (in our case, 2D coordinates)
        2. **Hidden Layers**: Process the information through weighted connections and activation functions
        3. **Output Layer**: Produces the final classification probabilities
        4. **Training**: The network adjusts its weights to minimize prediction errors using backpropagation
        """)
    
    with st.expander("What do the parameters mean?", expanded=False):
        st.markdown("""
        - **Hidden Layers**: More layers can learn more complex patterns but may overfit on small datasets
        - **Neurons per Layer**: More neurons increase the model's capacity to represent complex functions
        - **Activation Function**: 
          - **ReLU**: Fast and works well for most problems, but can "die" during training
          - **tanh**: Output between -1 and 1, good for centered data
          - **sigmoid**: Output between 0 and 1, can suffer from vanishing gradients
        - **Learning Rate**: Controls how big steps the model takes during learning
        - **Epochs**: Number of complete passes through the training data
        """)
    
    with st.expander("Interpreting the Results", expanded=False):
        st.markdown("""
        - **Decision Boundary**: The colored regions show how the MLP classifies different areas of the input space
        - **Training Loss**: Should generally decrease over time (if it doesn't, try adjusting the learning rate)
        - **Accuracy**: Percentage of correctly classified examples
        - **Overfitting**: If you use too many neurons/layers, the boundary might become too complex and not generalize well
        """)



if __name__ == "__main__":
    main()