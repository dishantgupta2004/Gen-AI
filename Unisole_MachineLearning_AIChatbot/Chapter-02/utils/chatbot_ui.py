import streamlit as st
import os
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import json
from pages.ai_tutor import MLPTutorBot
import pandas as pd

def create_chatbot_interface():
    """Create the chatbot interface in Streamlit."""
    
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'tutor_bot' not in st.session_state:
        st.session_state.tutor_bot = None
    
    # API Key input
    if st.session_state.tutor_bot is None:
        st.subheader("🤖 AI MLP Tutor")
        api_key = st.text_input(
            "Enter your OpenAI API Key to activate the AI tutor:",
            type="password",
            help="Your API key is not stored and only used for this session"
        )
        
        if api_key:
            try:
                st.session_state.tutor_bot = MLPTutorBot(api_key)
                st.success("AI Tutor activated! 🎉")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize AI tutor: {str(e)}")
        return
    
    # Chat interface
    st.subheader("🤖 Ask the AI MLP Tutor")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask me anything about MLPs, neural networks, or the current experiment...")
    
    if user_question:
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        
        # Get current page context
        current_page = st.session_state.get('current_page', None)
        user_level = st.session_state.get('user_level', 'beginner')
        experiment_results = st.session_state.get('last_experiment_results', None)
        
        context = {
            'page': current_page,
            'level': user_level,
            'results': experiment_results
        }
        
        # Get AI response
        with st.spinner("AI Tutor is thinking..."):
            try:
                ai_response = st.session_state.tutor_bot.get_contextual_response(user_question, context)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
    
    # Quick help buttons
    st.subheader("Quick Help")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💡 Explain Current Topic"):
            if current_page:
                topic_explanations = {
                    "1_what_is_MLP": "Explain what a multilayer perceptron is and how it works",
                    "2_feedforward": "Explain how feedforward computation works in neural networks",
                    "3_activation_functions": "Explain the different activation functions and their properties",
                    "4_weights_biases": "Explain the role of weights and biases in neural networks",
                    "5_backpropagation": "Explain how backpropagation and gradient descent work",
                    "6_overfitting_underfitting": "Explain overfitting, underfitting, and how to prevent them"
                }
                explanation_request = topic_explanations.get(current_page, "Explain the current topic")
                # Trigger AI response
                st.session_state.chat_messages.append({"role": "user", "content": explanation_request})
                st.rerun()
    
    with col2:
        if st.button("🎯 Suggest Next Steps"):
            if st.session_state.tutor_bot:
                suggestions = st.session_state.tutor_bot.suggest_next_steps(
                    current_page or "unknown", 
                    st.session_state.get('user_performance', {})
                )
                st.session_state.chat_messages.append({"role": "assistant", "content": suggestions})
                st.rerun()
    
    with col3:
        if st.button("❓ Common Questions"):
            common_questions = [
                "What's the difference between overfitting and underfitting?",
                "How do I choose the right activation function?",
                "Why is my neural network not learning?",
                "What learning rate should I use?",
                "How many hidden layers do I need?"
            ]
            selected_q = st.selectbox("Choose a common question:", [""] + common_questions)
            if selected_q:
                st.session_state.chat_messages.append({"role": "user", "content": selected_q})
                st.rerun()



