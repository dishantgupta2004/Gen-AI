"""
Document loaders for different input types (PDF, YouTube, URLs).
"""

import tempfile
import validators
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    YoutubeLoader, 
    UnstructuredURLLoader
)
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading documents from various sources."""
    
    @staticmethod
    def load_pdf(uploaded_file) -> List[Document]:
        """
        Load and parse PDF document from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
            
        Raises:
            Exception: If PDF loading fails
        """
        try:
            # Create temporary file to store uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "filename": uploaded_file.name,
                    "file_size": len(uploaded_file.getvalue())
                })
            
            logger.info(f"Successfully loaded PDF: {uploaded_file.name} ({len(documents)} pages)")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {uploaded_file.name}: {str(e)}")
            raise Exception(f"Failed to load PDF: {str(e)}")
    
    @staticmethod
    def load_youtube(url: str) -> List[Document]:
        """
        Load and parse YouTube video transcript.
        
        Args:
            url: YouTube video URL
            
        Returns:
            List of Document objects
            
        Raises:
            Exception: If YouTube loading fails
        """
        try:
            if not DocumentLoader._is_youtube_url(url):
                raise ValueError("Invalid YouTube URL format")
            
            # Load YouTube transcript
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True,
                language=["en", "en-US"],
                translation="en"
            )
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "youtube",
                    "url": url
                })
            
            logger.info(f"Successfully loaded YouTube video: {url}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading YouTube video {url}: {str(e)}")
            raise Exception(f"Failed to load YouTube video: {str(e)}")
    
    @staticmethod
    def load_url(url: str) -> List[Document]:
        """
        Load and parse content from web URL.
        
        Args:
            url: Web URL to load
            
        Returns:
            List of Document objects
            
        Raises:
            Exception: If URL loading fails
        """
        try:
            if not validators.url(url):
                raise ValueError("Invalid URL format")
            
            # Load URL content
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "url",
                    "url": url
                })
            
            logger.info(f"Successfully loaded URL: {url}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise Exception(f"Failed to load URL: {str(e)}")
    
    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        youtube_domains = ["youtube.com", "youtu.be", "m.youtube.com"]
        return any(domain in url.lower() for domain in youtube_domains)
    
    @staticmethod
    def validate_input(input_type: str, content) -> tuple[bool, str]:
        """
        Validate input based on type.
        
        Args:
            input_type: Type of input ("upload", "youtube", "url")
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if input_type == "upload":
            if content is None:
                return False, "Please upload a PDF file"
            if not content.name.lower().endswith('.pdf'):
                return False, "Only PDF files are supported"
            if content.size > 50 * 1024 * 1024:  # 50MB limit
                return False, "File size must be less than 50MB"
                
        elif input_type == "youtube":
            if not content or not content.strip():
                return False, "Please enter a YouTube URL"
            if not DocumentLoader._is_youtube_url(content):
                return False, "Please enter a valid YouTube URL"
                
        elif input_type == "url":
            if not content or not content.strip():
                return False, "Please enter a URL"
            if not validators.url(content):
                return False, "Please enter a valid URL"
        
        return True, ""

def load_documents(input_type: str, content) -> List[Document]:
    """
    Load documents based on input type and content.
    
    Args:
        input_type: Type of input ("upload", "youtube", "url") 
        content: Content to load (file object, URL string, etc.)
        
    Returns:
        List of Document objects
        
    Raises:
        Exception: If loading fails
    """
    loader = DocumentLoader()
    
    if input_type == "upload":
        return loader.load_pdf(content)
    elif input_type == "youtube":
        return loader.load_youtube(content)
    elif input_type == "url":
        return loader.load_url(content)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")