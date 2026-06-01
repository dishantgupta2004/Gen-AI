"""
Configuration settings for the Academic Paper Digest Generator.
"""

from dataclasses import dataclass
from typing import Dict, List
import streamlit as st
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@dataclass
class LLMConfig:
    """Configuration for LLM models and parameters."""
    
    # Groq Model configurations
    GROQ_MODELS = {
        "llama-3.1-70b-versatile": {
            "name": "Llama 3.1 70B",
            "description": "Most capable open-source model for complex reasoning",
            "max_tokens": 32768,
            "provider": "groq"
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B",
            "description": "Fast and efficient for quick summaries",
            "max_tokens": 32768,
            "provider": "groq"
        },
        "gemma2-9b-it": {
            "name": "Gemma 2 9B",
            "description": "Google's efficient model for academic content",
            "max_tokens": 8192,
            "provider": "groq"
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B",
            "description": "Mixture of experts model for detailed analysis",
            "max_tokens": 32768,
            "provider": "groq"
        }
    }
    
    # OpenAI Model configurations
    OPENAI_MODELS = {
        "gpt-4o": {
            "name": "GPT-4 Omni",
            "description": "Most capable OpenAI model for complex analysis",
            "max_tokens": 128000,
            "provider": "openai"
        },
        "gpt-4o-mini": {
            "name": "GPT-4 Omni Mini",
            "description": "Cost-effective and fast for academic summaries",
            "max_tokens": 128000,
            "provider": "openai"
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "description": "Reliable and affordable for most tasks",
            "max_tokens": 16384,
            "provider": "openai"
        }
    }
    
    # Combined models dictionary
    @classmethod
    def get_all_models(cls):
        all_models = {}
        all_models.update(cls.GROQ_MODELS)
        all_models.update(cls.OPENAI_MODELS)
        return all_models
    
    # Default model
    DEFAULT_MODEL = "llama-3.1-70b-versatile"
    
    # Generation parameters
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4000
    
    # Chunk sizes for different strategies
    CHUNK_SIZES = {
        "stuff": 16000,
        "map_reduce": 4000,
        "refine": 4000
    }

@dataclass
class AppConfig:
    """General application configuration."""
    
    APP_TITLE = "🎓 Academic Paper Digest Generator"
    PAGE_ICON = "🎓"
    LAYOUT = "wide"
    
    # File upload settings
    MAX_FILE_SIZE = 50  # MB
    ALLOWED_FORMATS = ["pdf"]
    
    # Supported input types
    INPUT_TYPES = {
        "PDF Upload": "upload",
        "YouTube Video": "youtube",
        "Research URL": "url"
    }
    
    # Summarization strategies
    SUMMARIZATION_STRATEGIES = {
        "Stuff": {
            "key": "stuff",
            "description": "Best for shorter documents. Processes entire content at once.",
            "recommended_for": "Papers < 20 pages",
            "explanation": "The 'Stuff' method takes all your document content and sends it to the AI model in a single request. It's the fastest method but has limitations on document size due to token limits."
        },
        "Map-Reduce": {
            "key": "map_reduce", 
            "description": "Ideal for longer documents. Summarizes chunks then combines.",
            "recommended_for": "Papers > 20 pages",
            "explanation": "Map-Reduce first breaks your document into smaller chunks, summarizes each chunk individually (Map phase), then combines all chunk summaries into a final comprehensive summary (Reduce phase)."
        },
        "Refine": {
            "key": "refine",
            "description": "Iteratively refines summary by processing chunks sequentially.",
            "recommended_for": "Complex, detailed analysis needed",
            "explanation": "Refine processes your document in chunks sequentially. It starts with the first chunk, creates a summary, then uses that summary plus the next chunk to create a refined summary, continuing until all content is processed."
        }
    }
    
    # Summary types
    SUMMARY_TYPES = {
        "Comprehensive": {
            "sections": ["abstract", "key_findings", "methodology", "future_work"],
            "description": "Complete analysis with all sections"
        },
        "Executive": {
            "sections": ["abstract", "key_findings"],
            "description": "High-level overview for quick understanding"
        },
        "Research Focus": {
            "sections": ["methodology", "key_findings", "future_work"],
            "description": "Focus on research methods and outcomes"
        }
    }

    # Educational content about summarization techniques
    SUMMARIZATION_EDUCATION = {
        "What is Text Summarization?": {
            "content": """
            Text summarization is the process of creating a shorter version of a text document while preserving its key information and main ideas. 
            It's like creating a highlight reel of the most important points from a longer piece of content.
            
            **Types of Summarization:**
            - **Extractive**: Selects and combines existing sentences from the original text
            - **Abstractive**: Generates new sentences that capture the essence of the original content
            - **Hybrid**: Combines both extractive and abstractive approaches
            """
        },
        "Why Use Different Strategies?": {
            "content": """
            Different documents require different approaches based on their length, complexity, and your specific needs:
            
            **Document Length Matters:**
            - Short documents (< 10 pages): Simple approaches work well
            - Medium documents (10-50 pages): Need chunking strategies
            - Long documents (> 50 pages): Require sophisticated processing
            
            **AI Model Limitations:**
            - Token limits: Models can only process a certain amount of text at once
            - Context windows: How much text the model can "remember" while processing
            - Quality vs. Speed: More sophisticated methods take longer but often produce better results
            """
        },
        "Understanding Our Strategies": {
            "content": """
            **🔹 Stuff Strategy:**
            - **How it works**: Puts entire document into a single AI request
            - **Best for**: Short papers, quick overviews
            - **Pros**: Fast, maintains context across entire document
            - **Cons**: Limited by model's token limit
            
            **🔹 Map-Reduce Strategy:**
            - **How it works**: Divides document → Summarizes each part → Combines summaries
            - **Best for**: Long documents, balanced speed/quality
            - **Pros**: Can handle very long documents, parallelizable
            - **Cons**: May lose some connections between sections
            
            **🔹 Refine Strategy:**
            - **How it works**: Builds summary incrementally, refining with each new section
            - **Best for**: When you need the most comprehensive analysis
            - **Pros**: Maintains context throughout, highest quality results
            - **Cons**: Slowest method, sequential processing required
            """
        },
        "Academic Summarization Tips": {
            "content": """
            **📚 For Academic Papers:**
            - Focus on research questions, methodology, findings, and implications
            - Preserve technical accuracy and scientific terminology
            - Maintain logical flow from problem → method → results → conclusions
            
            **🎯 Choosing Summary Types:**
            - **Comprehensive**: When you need full academic analysis
            - **Executive**: For quick decision-making or overviews
            - **Research Focus**: When methodology and results are most important
            
            **💡 Best Practices:**
            - Use higher temperature (0.3-0.7) for creative summarization
            - Use lower temperature (0.0-0.2) for factual, precise summaries
            - Consider your audience: technical experts vs. general readers
            - Always review AI-generated summaries for accuracy
            """
        },
        "Model Selection Guide": {
            "content": """
            **🚀 Groq Models (Fast & Free):**
            - **Llama 3.1 70B**: Best overall performance, complex reasoning
            - **Llama 3.1 8B**: Very fast, good for simple summaries
            - **Gemma 2 9B**: Excellent for academic content, Google-trained
            - **Mixtral 8x7B**: Great for detailed analysis, mixture of experts
            
            **🧠 OpenAI Models (Premium):**
            - **GPT-4 Omni**: Most capable, best for complex academic analysis
            - **GPT-4 Omni Mini**: Cost-effective, good balance of quality/price
            - **GPT-3.5 Turbo**: Reliable and affordable for most academic tasks
            
            **💰 Cost Considerations:**
            - Groq: Generally free with rate limits
            - OpenAI: Pay-per-token, varies by model
            - Consider document length when choosing models
            """
        }
    }

def get_page_config() -> Dict:
    """Get Streamlit page configuration."""
    return {
        "page_title": AppConfig.APP_TITLE,
        "page_icon": AppConfig.PAGE_ICON,
        "layout": AppConfig.LAYOUT,
        "initial_sidebar_state": "expanded"
    }