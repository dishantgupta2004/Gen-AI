import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

import openai

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Unified Q&A Chatbot"

# ------------------ PROMPT TEMPLATE ------------------ #
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# ------------------ RESPONSE FUNCTION ------------------ #
def generate_response(question, model_type, model_name, temperature, max_tokens, api_key=None):
    output_parser = StrOutputParser()
    
    if model_type == "OpenAI":
        openai.api_key = api_key
        llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "Ollama":
        llm = Ollama(model=model_name)
    else:
        raise ValueError("Invalid model type")

    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# ------------------ STREAMLIT UI ------------------ #
st.title("💬 Unified Q&A Chatbot (OpenAI + Ollama)")

model_type = st.sidebar.radio("Select Model Source", ["OpenAI", "Ollama"])

if model_type == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    model_name = st.sidebar.selectbox("OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
else:
    api_key = None  # not needed for Ollama
    model_name = st.sidebar.selectbox("Ollama Model", ["mistral", "llama3", "gemma"])

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

st.write("Ask your question below:")
user_input = st.text_input("You:")

if user_input:
    if model_type == "OpenAI" and not api_key:
        st.warning("Please enter your OpenAI API key.")
    else:
        response = generate_response(user_input, model_type, model_name, temperature, max_tokens, api_key)
        st.markdown(f"**Answer:** {response}")
else:
    st.info("Please enter a question to get started.")
