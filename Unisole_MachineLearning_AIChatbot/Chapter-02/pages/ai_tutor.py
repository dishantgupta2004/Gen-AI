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

class MLPTutorBot:
    def __init__(self, api_key: str):
        """Initialize the MLP Tutor Bot."""
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True
        )
        
        # Initialize knowledge base
        self.setup_knowledge_base()
        
        # Initialize conversation chain
        self.setup_conversation_chain()
        
        # Track current context
        self.current_page = None
        self.current_experiment_results = None
        self.user_level = "beginner"  # beginner, intermediate, advanced
    
    def setup_knowledge_base(self):
        """Create a knowledge base of MLP concepts for RAG."""
        
        # MLP Knowledge Base Content
        mlp_knowledge = [
            {
                "topic": "multilayer_perceptron_basics",
                "content": """
                A Multilayer Perceptron (MLP) is a feedforward artificial neural network consisting of:
                - Input layer: Receives the input features
                - Hidden layers: Process information through weighted connections
                - Output layer: Produces final predictions
                
                Key components:
                - Neurons/nodes: Basic processing units
                - Weights: Determine connection strength between neurons
                - Biases: Allow neurons to activate even with zero input
                - Activation functions: Introduce non-linearity (ReLU, sigmoid, tanh)
                """
            },
            {
                "topic": "feedforward_computation",
                "content": """
                Feedforward is the process of passing input through the network:
                1. Input layer receives data
                2. Each hidden layer computes: output = activation(weights × input + bias)
                3. Information flows forward to output layer
                4. Final layer produces predictions
                
                Mathematical formula:
                z = W × x + b (linear transformation)
                a = σ(z) (activation function)
                """
            },
            {
                "topic": "activation_functions",
                "content": """
                Activation functions introduce non-linearity:
                
                ReLU: f(x) = max(0, x)
                - Pros: Fast, helps with vanishing gradients
                - Cons: Dead neurons problem
                
                Sigmoid: f(x) = 1/(1 + e^(-x))
                - Pros: Output between 0-1, smooth
                - Cons: Vanishing gradients, not zero-centered
                
                Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
                - Pros: Zero-centered, smooth
                - Cons: Vanishing gradients
                """
            },
            {
                "topic": "backpropagation",
                "content": """
                Backpropagation is how neural networks learn:
                1. Forward pass: Compute predictions
                2. Calculate loss: Compare predictions to true values
                3. Backward pass: Compute gradients using chain rule
                4. Update weights: Use gradients to improve parameters
                
                Gradient descent: weights = weights - learning_rate × gradients
                Learning rate controls step size - too high causes instability, too low causes slow learning.
                """
            },
            {
                "topic": "overfitting_underfitting",
                "content": """
                Overfitting: Model memorizes training data but fails to generalize
                - Signs: Training accuracy >> Validation accuracy
                - Solutions: More data, regularization, early stopping, simpler model
                
                Underfitting: Model too simple to capture patterns
                - Signs: Both training and validation accuracy are low
                - Solutions: More complex model, better features, less regularization
                
                Goal: Find balance between bias (underfitting) and variance (overfitting)
                """
            }
        ]
        
        # Convert to documents
        documents = []
        for item in mlp_knowledge:
            doc = Document(
                page_content=item["content"],
                metadata={"topic": item["topic"]}
            )
            documents.append(doc)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.knowledge_base = FAISS.from_documents(documents, embeddings)
    
    def setup_conversation_chain(self):
        """Setup the conversation chain with system prompt."""
        
        system_template = """
        You are an expert AI tutor for neural networks and machine learning, specifically focused on Multilayer Perceptrons (MLPs). 
        
        Your role:
        - Explain MLP concepts clearly and pedagogically
        - Provide step-by-step guidance for learning
        - Answer questions about neural network theory and practice
        - Help interpret experimental results and visualizations
        - Suggest next learning steps based on user progress
        - Adapt explanations to user's experience level: {user_level}
        
        Current context:
        - User is on page: {current_page}
        - Recent experiment results: {experiment_results}
        
        Guidelines:
        - Use simple language for beginners, technical details for advanced users
        - Always relate concepts to practical examples
        - Encourage hands-on experimentation
        - Ask clarifying questions when needed
        - Provide actionable advice
        - Use emojis sparingly but effectively for engagement
        
        Remember: You're helping users understand MLPs through an interactive learning app.
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=chat_prompt,
            verbose=False
        )
    
    def get_contextual_response(self, user_question: str, context: Dict = None) -> str:
        """Get AI response with context awareness."""
        
        # Update context if provided
        if context:
            self.current_page = context.get('page', self.current_page)
            self.current_experiment_results = context.get('results', self.current_experiment_results)
            self.user_level = context.get('level', self.user_level)
        
        # Search knowledge base for relevant information
        relevant_docs = self.knowledge_base.similarity_search(user_question, k=2)
        knowledge_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Enhance question with context
        enhanced_question = f"""
        Question: {user_question}
        
        Relevant knowledge: {knowledge_context}
        
        Current page: {self.current_page or 'Unknown'}
        Recent results: {self.current_experiment_results or 'None'}
        User level: {self.user_level}
        """
        
        # Get response from conversation chain
        try:
            response = self.conversation.predict(
                input=enhanced_question,
                user_level=self.user_level,
                current_page=self.current_page or 'Unknown',
                experiment_results=str(self.current_experiment_results) if self.current_experiment_results else 'None'
            )
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Could you please rephrase your question?"
    
    def suggest_next_steps(self, current_page: str, user_performance: Dict) -> str:
        """Suggest next learning steps based on progress."""
        
        suggestions = {
            "1_what_is_MLP": [
                "Try different activation functions to see how they affect the decision boundary",
                "Experiment with varying the number of hidden layers",
                "Move to the Feedforward page to understand the mathematical details"
            ],
            "2_feedforward": [
                "Practice with different network architectures",
                "Try the Activation Functions page to see how different functions behave",
                "Experiment with manual weight adjustments"
            ],
            "3_activation_functions": [
                "Compare how different activations perform on the same dataset",
                "Move to Weights & Biases to understand parameter effects",
                "Try the interactive function explorer with extreme values"
            ],
            "4_weights_biases": [
                "Experiment with different weight initialization methods",
                "Try the Backpropagation page to see how weights are updated",
                "Test how bias affects the decision boundary position"
            ],
            "5_backpropagation": [
                "Experiment with different learning rates",
                "Try the Overfitting page to see training dynamics",
                "Compare different optimizers"
            ],
            "6_overfitting_underfitting": [
                "Try all three experiment types to build intuition",
                "Experiment with regularization strengths",
                "Go back to earlier pages and try more complex scenarios"
            ]
        }
        
        page_suggestions = suggestions.get(current_page, ["Continue exploring the interactive features!"])
        return "Here are some suggested next steps:\n" + "\n".join([f"• {s}" for s in page_suggestions])

