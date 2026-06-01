
import streamlit as st
from typing import Dict, Any, Optional, Callable
from app.config import AppConfig, LLMConfig
import plotly.graph_objects as go
import plotly.express as px

class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def render_header():
        """Render the main application header."""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">
                🎓 Academic Paper Digest Generator
            </h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Transform research papers, videos, and articles into structured academic summaries
            </p>
        </div>
        """, unsafe_allow_html=True)
    
        # Quick reference card
        st.markdown("---")
        st.subheader("🎯 Quick Reference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 Document Size Guide:**
            - Short (< 10 pages): Use Stuff
            - Medium (10-50 pages): Use Map-Reduce  
            - Long (> 50 pages): Use Refine
            """)
            
        with col2:
            st.markdown("""
            **🤖 Model Recommendations:**
            - Complex Analysis: Llama 3.1 70B, GPT-4 Omni
            - Quick Summaries: Llama 3.1 8B, GPT-4 Mini
            - Academic Focus: Gemma 2 9B
            """)
        
        # Tips section
        st.info("""
        💡 **Pro Tips:**
        - Start with free Groq models to experiment
        - Use lower temperature (0.0-0.2) for factual summaries
        - Use higher temperature (0.3-0.7) for creative analysis
        - Always review AI summaries for accuracy
        """)
    
    @staticmethod
    def render_input_selector() -> tuple[str, Any]:
        """
        Render input type selector and corresponding input widget.
        
        Returns:
            Tuple of (input_type, content)
        """
        st.subheader("📄 Select Input Source")
        
        # Input type selection
        input_type_name = st.selectbox(
            "Choose input type:",
            options=list(AppConfig.INPUT_TYPES.keys()),
            help="Select the type of content you want to summarize"
        )
        
        input_type = AppConfig.INPUT_TYPES[input_type_name]
        content = None
        
        # Render appropriate input widget based on type
        if input_type == "upload":
            content = st.file_uploader(
                "Upload PDF file",
                type=AppConfig.ALLOWED_FORMATS,
                help=f"Maximum file size: {AppConfig.MAX_FILE_SIZE}MB"
            )
            
            if content:
                st.success(f"✅ Uploaded: {content.name} ({content.size:,} bytes)")
                
        elif input_type == "youtube":
            content = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=example",
                help="Enter a YouTube video URL to summarize its transcript"
            )
            
        elif input_type == "url":
            content = st.text_input(
                "Research URL",
                placeholder="https://arxiv.org/abs/example or research article URL",
                help="Enter URL of a research paper or academic article"
            )
        
        return input_type, content
    
    @staticmethod
    def render_summarization_options() -> Dict[str, str]:
        """
        Render summarization strategy and type selection.
        
        Returns:
            Dictionary with selected options
        """
        st.subheader("🎯 Summarization Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strategy Selection**")
            strategy_name = st.selectbox(
                "Choose strategy:",
                options=list(AppConfig.SUMMARIZATION_STRATEGIES.keys()),
                help="Select the summarization approach"
            )
            
            strategy_info = AppConfig.SUMMARIZATION_STRATEGIES[strategy_name]
            st.info(f"ℹ️ {strategy_info['description']}")
            st.caption(f"**Recommended for:** {strategy_info['recommended_for']}")
            
            # Show detailed explanation in expander
            with st.expander("🔍 How this strategy works"):
                st.markdown(strategy_info['explanation'])
        
        with col2:
            st.markdown("**Summary Type Selection**")
            summary_type = st.selectbox(
                "Choose summary type:",
                options=list(AppConfig.SUMMARY_TYPES.keys()),
                help="Select the type of analysis you need"
            )
            
            summary_info = AppConfig.SUMMARY_TYPES[summary_type]
            st.info(f"ℹ️ {summary_info['description']}")
            st.caption(f"**Includes:** {', '.join(summary_info['sections'])}")
        
        return {
            "strategy": AppConfig.SUMMARIZATION_STRATEGIES[strategy_name]["key"],
            "summary_type": summary_type
        }
    
    @staticmethod
    def render_progress_indicator(message: str = "Processing..."):
        """Render a progress indicator with message."""
        progress_placeholder = st.empty()
        
        def update_progress(new_message: str):
            progress_placeholder.info(f"⏳ {new_message}")
        
        update_progress(message)
        return update_progress
    
    @staticmethod
    def render_summary_output(summary: str, metadata: Dict[str, Any] = None):
        """
        Render the generated summary with formatting and metadata.
        
        Args:
            summary: Generated summary text
            metadata: Optional metadata about the summarization
        """
        st.subheader("📋 Generated Summary")
        
        # Display metadata if provided
        if metadata:
            with st.expander("📊 Summary Metadata", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "word_count" in metadata:
                        st.metric("Word Count", metadata["word_count"])
                    if "source_type" in metadata:
                        st.metric("Source Type", metadata["source_type"].upper())
                
                with col2:
                    if "strategy" in metadata:
                        st.metric("Strategy", metadata["strategy"].title())
                    if "model" in metadata:
                        st.metric("Model", metadata["model"])
                
                with col3:
                    if "processing_time" in metadata:
                        st.metric("Processing Time", f"{metadata['processing_time']:.1f}s")
                    if "estimated_tokens" in metadata:
                        st.metric("Est. Tokens", f"{metadata['estimated_tokens']:,}")
        
        # Display summary in a styled container
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
        """, unsafe_allow_html=True)
        
        st.markdown(summary)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="academic_summary.md",
            mime="text/markdown",
            help="Download the summary as a markdown file"
        )
    
    @staticmethod
    def render_error_message(error: str, error_type: str = "Error"):
        """Render an error message with appropriate styling."""
        st.error(f"❌ **{error_type}:** {error}")
    
    @staticmethod
    def render_success_message(message: str):
        """Render a success message."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def render_info_message(message: str):
        """Render an info message."""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def render_document_stats(documents, estimated_tokens: int = None):
        """
        Render statistics about loaded documents.
        
        Args:
            documents: List of loaded documents
            estimated_tokens: Estimated token count
        """
        if not documents:
            return
        
        st.subheader("📊 Document Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", len(documents))
        
        with col2:
            total_chars = sum(len(doc.page_content) for doc in documents)
            st.metric("Characters", f"{total_chars:,}")
        
        with col3:
            total_words = sum(len(doc.page_content.split()) for doc in documents)
            st.metric("Words", f"{total_words:,}")
        
        with col4:
            if estimated_tokens:
                st.metric("Est. Tokens", f"{estimated_tokens:,}")
        
        # Document details in expander
        with st.expander("📄 Document Details", expanded=False):
            for i, doc in enumerate(documents):
                st.markdown(f"**Document {i+1}:**")
                st.markdown(f"- Characters: {len(doc.page_content):,}")
                st.markdown(f"- Words: {len(doc.page_content.split()):,}")
                if doc.metadata:
                    st.markdown(f"- Metadata: {doc.metadata}")
                st.markdown("---")
    
    @staticmethod
    def render_strategy_recommendation(documents, recommended_strategy: str):
        """
        Render strategy recommendation based on document analysis.
        
        Args:
            documents: List of documents
            recommended_strategy: Recommended strategy name
        """
        st.info(f"💡 **Recommended Strategy:** {recommended_strategy.title()}")
        
        total_length = sum(len(doc.page_content) for doc in documents)
        
        if total_length < 10000:
            explanation = "Document is relatively short - stuff strategy will work efficiently."
        elif total_length < 50000:
            explanation = "Medium-length document - map-reduce will handle this well."
        else:
            explanation = "Long document - refine strategy recommended for thorough analysis."
        
        st.caption(explanation)
    
    @staticmethod
    def render_footer():
        """Render application footer."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Built with ❤️ using Streamlit, LangChain, and Anthropic Claude</p>
            <p style="font-size: 0.8rem;">
                For best results, ensure your API key has sufficient credits and 
                your documents are well-formatted academic content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    
    @staticmethod
    def render_sidebar_config() -> Dict[str, Any]:
        """
        Render sidebar configuration options.
        Returns:
        Dictionary with configuration values
        """
        # Create tabs for different sections
        tab1, tab2 = st.sidebar.tabs(["⚙️ Config", "📚 Learn"])
        
        with tab1:
            st.header("Configuration")
            # Provider selection
            provider = st.selectbox(
                "Select AI Provider",
                options=["groq", "openai"],
                format_func=lambda x: "🚀 Groq (Free/Fast)" if x == "groq" else "🧠 OpenAI (Premium)",
                help="Choose between Groq (free with rate limits) or OpenAI (paid)"
            )
            
            # API Key input with provider-specific placeholder
            api_key_placeholder = "gsk_..." if provider == "groq" else "sk-..."
            api_key_label = "Groq API Key" if provider == "groq" else "OpenAI API Key"
            
            api_key = st.text_input(
                api_key_label,
                type="password",
                help=f"Enter your {provider.upper()} API key",
                placeholder=api_key_placeholder
            )
            
            # Model selection based on provider
            all_models = LLMConfig.get_all_models()
            provider_models = {k: v for k, v in all_models.items() if v.get("provider") == provider}
            
            selected_model = st.selectbox(
                "Select Model",
                options=list(provider_models.keys()),
                format_func=lambda x: provider_models[x]["name"],
                help="Choose the AI model for summarization"
            )
            
            # Display model info
            model_info = provider_models[selected_model]
            st.info(f"ℹ️ {model_info['description']}")
            
            # Advanced settings expander
            with st.expander("🔧 Advanced Settings"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=LLMConfig.DEFAULT_TEMPERATURE,
                    step=0.1,
                    help="Controls randomness. Lower = more focused, Higher = more creative."
                )
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1000,
                    max_value=8000,
                    value=LLMConfig.DEFAULT_MAX_TOKENS,
                    step=500,
                    help="Maximum tokens for summary generation"
                )
            
                # Provider-specific information
            if provider == "groq":
                st.success("✅ Groq offers fast, free inference with rate limits")
                st.caption("Get your free API key at: console.groq.com")
            else:
                st.info("💰 OpenAI charges per token used")
                st.caption("Get your API key at: platform.openai.com")
        
        with tab2:
            UIComponents.render_educational_content()
        
        return {
            "api_key": api_key,
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": provider
        }
    
    @staticmethod
    def render_educational_content():
        """Render educational content about text summarization."""
        st.header("📚 Text Summarization Guide")
        
        # Create expandable sections for each educational topic
        education_content = AppConfig.SUMMARIZATION_EDUCATION
        
        for topic, info in education_content.items():
            with st.expander(f"📖 {topic}", expanded=False):
                st.markdown(info["content"])
        
        # Quick reference card
        st.markdown("---")

class ValidationHelpers:
    """Helper functions for input validation and user feedback."""
    
    @staticmethod
    def validate_api_key(api_key: str, provider: str) -> tuple[bool, str]:
        """Validate API key format based on provider."""
        if not api_key:
            return False, "API key is required"
        
        if provider == "groq":
            if not api_key.startswith("gsk_"):
                return False, "Invalid Groq API key format. Should start with 'gsk_'"
        elif provider == "openai":
            if not api_key.startswith("sk-"):
                return False, "Invalid OpenAI API key format. Should start with 'sk-'"
        
        if len(api_key) < 20:
            return False, "API key appears to be too short"
        return True, ""
    
    @staticmethod
    def validate_inputs(config: Dict[str, Any], input_type: str, content: Any) -> tuple[bool, str]:
        """
        Validate all required inputs.
        
        Args:
            config: Configuration dictionary
            input_type: Type of input
            content: Input content
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate API key
        api_valid, api_error = ValidationHelpers.validate_api_key(
            config.get("api_key", ""), 
            config.get("provider", "groq")
        )
        if not api_valid:
            return False, api_error
        
        # Validate content based on input type
        if input_type == "upload":
            if content is None:
                return False, "Please upload a PDF file"
        elif input_type in ["youtube", "url"]:
            if not content or not content.strip():
                return False, f"Please enter a {'YouTube URL' if input_type == 'youtube' else 'URL'}"
        
        return True, ""