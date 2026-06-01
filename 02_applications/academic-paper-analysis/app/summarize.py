"""
Main summarization logic and workflow orchestration.
"""

import time
import logging
from typing import Dict, Any, List
from langchain.schema import Document

from utils.loader import load_documents, DocumentLoader
from utils.summarizer import create_summarizer
from components.ui_components import UIComponents, ValidationHelpers
import streamlit as st
from app.config import LLMConfig
logger = logging.getLogger(__name__)

class SummarizationWorkflow:
    """Orchestrates the complete summarization workflow."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.summary: str = ""
        self.metadata: Dict[str, Any] = {}
    
    def execute_workflow(
        self, 
        config: Dict[str, Any], 
        input_type: str, 
        content: Any,
        summarization_options: Dict[str, str]
    ) -> tuple[bool, str]:
        """
        Execute the complete summarization workflow.
        
        Args:
            config: Configuration dictionary with API key, model, etc.
            input_type: Type of input ("upload", "youtube", "url")
            content: Input content (file, URL, etc.)
            summarization_options: Strategy and type selections
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Step 1: Validate inputs
            is_valid, error_msg = ValidationHelpers.validate_inputs(config, input_type, content)
            if not is_valid:
                return False, error_msg
            
            # Step 2: Load documents
            success = self._load_documents(input_type, content)
            if not success:
                return False, "Failed to load documents"
            
            # Step 3: Generate summary
            success = self._generate_summary(config, summarization_options)
            if not success:
                return False, "Failed to generate summary"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return False, f"Workflow failed: {str(e)}"
    
    def _load_documents(self, input_type: str, content: Any) -> bool:
        """Load documents from the specified input."""
        try:
            with st.spinner("📄 Loading documents..."):
                # Validate input first
                is_valid, error_msg = DocumentLoader.validate_input(input_type, content)
                if not is_valid:
                    UIComponents.render_error_message(error_msg, "Validation Error")
                    return False
                
                # Load documents
                self.documents = load_documents(input_type, content)
                
                if not self.documents:
                    UIComponents.render_error_message("No content could be extracted from the input")
                    return False
                
                # Display document statistics
                UIComponents.render_success_message(f"Successfully loaded {len(self.documents)} document(s)")
                
                # Create summarizer to get token estimate (with dummy key for estimation only)
                try:
                    temp_summarizer = create_summarizer(
                        api_key='dummy_key_for_estimation',
                        model=LLMConfig.DEFAULT_MODEL
                    )
                    estimated_tokens = temp_summarizer.estimate_tokens(self.documents)
                except:
                    # Fallback calculation if summarizer creation fails
                    total_text = "\n".join([doc.page_content for doc in self.documents])
                    estimated_tokens = len(total_text) // 4
                
                UIComponents.render_document_stats(self.documents, estimated_tokens)
                
                try:
                    temp_summarizer = create_summarizer(
                        api_key='dummy_key_for_recommendation',
                        model=LLMConfig.DEFAULT_MODEL
                    )
                    recommended_strategy = temp_summarizer.get_recommended_strategy(self.documents)
                except:
                    # Fallback recommendation based on simple text length
                    total_length = sum(len(doc.page_content) for doc in self.documents)
                    if total_length < 10000:
                        recommended_strategy = "stuff"
                    elif total_length < 50000:
                        recommended_strategy = "map_reduce"
                    else:
                        recommended_strategy = "refine"
                UIComponents.render_strategy_recommendation(self.documents, recommended_strategy)
                
                return True
                
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            UIComponents.render_error_message(str(e), "Loading Error")
            return False
    
    def _generate_summary(self, config: Dict[str, Any], options: Dict[str, str]) -> bool:
        """Generate summary using the specified configuration and options."""
        try:
            start_time = time.time()
            
            # Initialize progress callback
            progress_update = UIComponents.render_progress_indicator("Initializing summarization...")
            
            # Create summarizer with provider information
            summarizer = create_summarizer(
                api_key=config["api_key"],
                model=config["model"]
            )
            
            # Generate summary
            self.summary = summarizer.summarize(
                documents=self.documents,
                strategy=options["strategy"],
                summary_type=options["summary_type"],
                progress_callback=progress_update
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare metadata
            self.metadata = {
                "word_count": len(self.summary.split()),
                "source_type": self.documents[0].metadata.get("source_type", "unknown"),
                "strategy": options["strategy"],
                "summary_type": options["summary_type"],
                "model": config["model"],
                "provider": config.get("provider", "groq"),
                "processing_time": processing_time,
                "estimated_tokens": summarizer.estimate_tokens(self.documents),
                "document_count": len(self.documents)
            }
            
            # Clear progress indicator
            progress_update("Summary generation completed!")
            
            return True
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            UIComponents.render_error_message(str(e), "Summarization Error")
            return False
    
    def display_results(self):
        """Display the summarization results."""
        if self.summary:
            UIComponents.render_summary_output(self.summary, self.metadata)
        else:
            UIComponents.render_error_message("No summary available to display")

def run_summarization_workflow():
    """Main function to run the summarization workflow in Streamlit."""
    
    # Initialize workflow
    workflow = SummarizationWorkflow()
    
    # Get configuration from sidebar
    config = UIComponents.render_sidebar_config()
    
    # Store API key temporarily for document loading validation
    st.session_state['temp_api_key'] = config.get("api_key", "")
    
    # Get input selection
    input_type, content = UIComponents.render_input_selector()
    
    # Get summarization options
    summarization_options = UIComponents.render_summarization_options()
    
    # Main action button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Generate Academic Summary",
            type="primary",
            use_container_width=True,
            help="Start the summarization process"
        ):
            # Execute workflow
            success, error_msg = workflow.execute_workflow(
                config, input_type, content, summarization_options
            )
            
            if success:
                # Display results
                st.markdown("---")
                workflow.display_results()
            else:
                if error_msg:
                    UIComponents.render_error_message(error_msg)
    
    # Display usage tips
    with st.expander("💡 Usage Tips", expanded=False):
        st.markdown("""
        **For best results:**
        
        📄 **PDF Files:**
        - Use well-formatted academic papers
        - Ensure text is selectable (not scanned images)
        - Keep files under 50MB
        
        🎥 **YouTube Videos:**
        - Works best with educational/research content
        - Requires videos to have captions/transcripts
        - Academic lectures and talks are ideal
        
        🔗 **Research URLs:**
        - Use direct links to papers (arXiv, journals, etc.)
        - Avoid pages requiring authentication
        - Works well with academic repositories
        
        ⚙️ **Strategy Selection:**
        - **Stuff**: Fast, good for short papers (< 20 pages)
        - **Map-Reduce**: Best for longer documents (> 20 pages)
        - **Refine**: Most thorough, slower but detailed analysis
        """)

# Session state management for workflow
def initialize_session_state():
    """Initialize session state variables."""
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None