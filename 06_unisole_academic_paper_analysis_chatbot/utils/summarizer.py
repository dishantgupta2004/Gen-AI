"""
LLM summarization chains and orchestration logic.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from utils.prompts import (
    get_prompt_by_type, 
    AcademicPrompts
)
from app.config import LLMConfig
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class AcademicSummarizer:
    """Handles academic content summarization using different strategies."""
    
    def __init__(self, api_key: str, model: str = None, temperature: float = None, provider: str = "groq"):
        """
        Initialize the summarizer with LLM configuration.
        
        Args:
            api_key: API key (Groq or OpenAI)
            model: Model name to use
            temperature: Temperature for generation
            provider: LLM provider ("groq" or "openai")
        """
        self.api_key = api_key
        self.model = model or LLMConfig.DEFAULT_MODEL
        self.temperature = temperature or LLMConfig.DEFAULT_TEMPERATURE
        self.provider = provider
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=LLMConfig.CHUNK_SIZES["stuff"],
            chunk_overlap=200,
            length_function=len,
        )
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider."""
        try:
            # Skip actual LLM initialization for dummy keys (used for estimation only)
            if 'dummy' in self.api_key.lower():
                return None
                
            all_models = LLMConfig.get_all_models()
            model_info = all_models.get(self.model, {})
            provider = model_info.get("provider", self.provider)
            
            if provider == "groq":
                return ChatGroq(
                    groq_api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=LLMConfig.DEFAULT_MAX_TOKENS
                )
            elif provider == "openai":
                return ChatOpenAI(
                    openai_api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=LLMConfig.DEFAULT_MAX_TOKENS
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            # If this is a dummy key, just return None
            if 'dummy' in self.api_key.lower():
                return None
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")
    
    def summarize(
        self, 
        documents: List[Document], 
        strategy: str, 
        summary_type: str,
        progress_callback=None
    ) -> str:
        """
        Summarize documents using specified strategy and type.
        
        Args:
            documents: List of documents to summarize
            strategy: Summarization strategy ("stuff", "map_reduce", "refine")
            summary_type: Type of summary ("Comprehensive", "Executive", "Research Focus")
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated summary text
        """
        try:
            # Check if this is a dummy summarizer (used for estimation only)
            if self.llm is None:
                raise Exception("Cannot generate summary with dummy API key")
                
            if progress_callback:
                progress_callback("Initializing summarization...")
            
            # Get appropriate prompt for summary type
            prompt = get_prompt_by_type(summary_type)
            
            if strategy == "stuff":
                return self._stuff_summarize(documents, prompt, progress_callback)
            elif strategy == "map_reduce":
                return self._map_reduce_summarize(documents, prompt, progress_callback)
            elif strategy == "refine":
                return self._refine_summarize(documents, prompt, progress_callback)
            else:
                raise ValueError(f"Unsupported summarization strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise Exception(f"Summarization failed: {str(e)}")
    
    def _stuff_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using stuff strategy."""
        try:
            if progress_callback:
                progress_callback("Processing documents with stuff strategy...")
            
            # Combine all documents
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Check if content is too long
            if len(combined_text) > LLMConfig.CHUNK_SIZES["stuff"]:
                logger.warning("Content may be too long for stuff strategy")
                # Truncate if necessary
                combined_text = combined_text[:LLMConfig.CHUNK_SIZES["stuff"]]
            
            # Create chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            if progress_callback:
                progress_callback("Generating summary...")
            
            # Generate summary
            result = chain.run(documents)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Stuff summarization failed: {str(e)}")
            raise Exception(f"Stuff summarization failed: {str(e)}")
    
    def _map_reduce_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using map-reduce strategy."""
        try:
            if progress_callback:
                progress_callback("Splitting documents for map-reduce...")
            
            # Split documents into smaller chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))
            
            if progress_callback:
                progress_callback(f"Processing {len(chunks)} chunks...")
            
            # Create map-reduce chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=AcademicPrompts.get_map_reduce_prompt(),
                combine_prompt=prompt,
                verbose=True
            )
            
            if progress_callback:
                progress_callback("Generating summary...")
            
            # Generate summary
            result = chain.run(chunks)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Map-reduce summarization failed: {str(e)}")
            raise Exception(f"Map-reduce summarization failed: {str(e)}")
    
    def _refine_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using refine strategy."""
        try:
            if progress_callback:
                progress_callback("Preparing documents for refine strategy...")
            
            # Split documents into smaller chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))
            
            if progress_callback:
                progress_callback(f"Refining summary through {len(chunks)} iterations...")
            
            # Create refine chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=AcademicPrompts.get_refine_prompt(),
                verbose=True
            )
            
            if progress_callback:
                progress_callback("Generating refined summary...")
            
            # Generate summary
            result = chain.run(chunks)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Refine summarization failed: {str(e)}")
            raise Exception(f"Refine summarization failed: {str(e)}")
    
    def estimate_tokens(self, documents: List[Document]) -> int:
        """Estimate token count for documents."""
        total_text = "\n".join([doc.page_content for doc in documents])
        # Rough estimation: 1 token ≈ 4 characters
        return len(total_text) // 4
    
    def get_recommended_strategy(self, documents: List[Document]) -> str:
        """Recommend summarization strategy based on document size."""
        total_length = sum(len(doc.page_content) for doc in documents)
        
        if total_length < 10000:  # ~2500 tokens
            return "stuff"
        elif total_length < 50000:  # ~12500 tokens
            return "map_reduce"
        else:
            return "refine"

def create_summarizer(api_key: str, model: str = None) -> AcademicSummarizer:
    """
    Create and return an AcademicSummarizer instance.
    
    Args:
        api_key: API key (Groq or OpenAI)
        model: Optional model name
        
    Returns:
        AcademicSummarizer instance
    """
    # Auto-detect provider based on model
    if model is None:
        model = LLMConfig.DEFAULT_MODEL
    
    all_models = LLMConfig.get_all_models()
    if model in all_models:
        provider = all_models[model].get("provider", "groq")
    else:
        provider = "groq"  # Default to groq
    
    return AcademicSummarizer(api_key=api_key, model=model, provider=provider)
    
    def summarize(
        self, 
        documents: List[Document], 
        strategy: str, 
        summary_type: str,
        progress_callback=None
    ) -> str:
        """
        Summarize documents using specified strategy and type.
        
        Args:
            documents: List of documents to summarize
            strategy: Summarization strategy ("stuff", "map_reduce", "refine")
            summary_type: Type of summary ("Comprehensive", "Executive", "Research Focus")
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated summary text
        """
        try:
            if progress_callback:
                progress_callback("Initializing summarization...")
            
            # Get appropriate prompt for summary type
            prompt = get_prompt_by_type(summary_type)
            
            if strategy == "stuff":
                return self._stuff_summarize(documents, prompt, progress_callback)
            elif strategy == "map_reduce":
                return self._map_reduce_summarize(documents, prompt, progress_callback)
            elif strategy == "refine":
                return self._refine_summarize(documents, prompt, progress_callback)
            else:
                raise ValueError(f"Unsupported summarization strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise Exception(f"Summarization failed: {str(e)}")
    
    def _stuff_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using stuff strategy."""
        try:
            if progress_callback:
                progress_callback("Processing documents with stuff strategy...")
            
            # Combine all documents
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Check if content is too long
            if len(combined_text) > LLMConfig.CHUNK_SIZES["stuff"]:
                logger.warning("Content may be too long for stuff strategy")
                # Truncate if necessary
                combined_text = combined_text[:LLMConfig.CHUNK_SIZES["stuff"]]
            
            # Create chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            if progress_callback:
                progress_callback("Generating summary...")
            
            # Generate summary
            result = chain.run(documents)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Stuff summarization failed: {str(e)}")
            raise Exception(f"Stuff summarization failed: {str(e)}")
    
    def _map_reduce_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using map-reduce strategy."""
        try:
            if progress_callback:
                progress_callback("Splitting documents for map-reduce...")
            
            # Split documents into smaller chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))
            
            if progress_callback:
                progress_callback(f"Processing {len(chunks)} chunks...")
            
            # Create map-reduce chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=AcademicPrompts.get_map_reduce_prompt(),
                combine_prompt=prompt,
                verbose=True
            )
            
            if progress_callback:
                progress_callback("Generating summary...")
            
            # Generate summary
            result = chain.run(chunks)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Map-reduce summarization failed: {str(e)}")
            raise Exception(f"Map-reduce summarization failed: {str(e)}")
    
    def _refine_summarize(
        self, 
        documents: List[Document], 
        prompt, 
        progress_callback=None
    ) -> str:
        """Summarize using refine strategy."""
        try:
            if progress_callback:
                progress_callback("Preparing documents for refine strategy...")
            
            # Split documents into smaller chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))
            
            if progress_callback:
                progress_callback(f"Refining summary through {len(chunks)} iterations...")
            
            # Create refine chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=AcademicPrompts.get_refine_prompt(),
                verbose=True
            )
            
            if progress_callback:
                progress_callback("Generating refined summary...")
            
            # Generate summary
            result = chain.run(chunks)
            
            if progress_callback:
                progress_callback("Summary completed!")
            
            return result
            
        except Exception as e:
            logger.error(f"Refine summarization failed: {str(e)}")
            raise Exception(f"Refine summarization failed: {str(e)}")
    
    def estimate_tokens(self, documents: List[Document]) -> int:
        """Estimate token count for documents."""
        total_text = "\n".join([doc.page_content for doc in documents])
        # Rough estimation: 1 token ≈ 4 characters
        return len(total_text) // 4
    
    def get_recommended_strategy(self, documents: List[Document]) -> str:
        """Recommend summarization strategy based on document size."""
        total_length = sum(len(doc.page_content) for doc in documents)
        
        if total_length < 10000:  # ~2500 tokens
            return "stuff"
        elif total_length < 50000:  # ~12500 tokens
            return "map_reduce"
        else:
            return "refine"

def create_summarizer(api_key: str, model: str = None) -> AcademicSummarizer:
    """
    Create and return an AcademicSummarizer instance.
    
    Args:
        api_key: Anthropic API key
        model: Optional model name
        
    Returns:
        AcademicSummarizer instance
    """
    return AcademicSummarizer(api_key=api_key, model=model)