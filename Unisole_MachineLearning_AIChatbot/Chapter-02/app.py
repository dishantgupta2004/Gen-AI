"""
Main entry point for the MLP Learning Streamlit Application.
This app teaches users about multilayer perceptrons through interactive demonstrations.
"""

import streamlit as st
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

def main():
    """Main function to set up the Streamlit app configuration."""
    st.set_page_config(
        page_title="MLP Learning App",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main page content
    st.title("🧠 Multilayer Perceptron Learning App")
    st.markdown("""
    Welcome to the interactive MLP learning application! This app will guide you through
    the fundamentals of multilayer perceptrons (neural networks) with hands-on examples.
    
    ## What you'll learn:
    - **What is an MLP?** - Basic concepts and interactive classification
    - **Feedforward Computation** - How data flows through the network
    - **Activation Functions** - ReLU, tanh, and sigmoid comparisons
    - **Weights and Biases** - Understanding network parameters
    - **Backpropagation** - How networks learn from mistakes
    - **Overfitting vs Underfitting** - Model complexity trade-offs
    
    👈 **Select a topic from the sidebar to get started!**
    """)
    
    st.info("💡 Tip: Adjust the parameters in the sidebar on each page to see how they affect the model's behavior!")

if __name__ == "__main__":
    main()