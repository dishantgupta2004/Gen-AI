"""
Main Streamlit application for the Academic Paper Digest Generator.
"""

import streamlit as st
import logging
from config import get_page_config
from summarize import run_summarization_workflow, initialize_session_state
from components.ui_components import UIComponents
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    
    # Set page configuration
    st.set_page_config(**get_page_config())
    
    # Initialize session state
    initialize_session_state()
    
    try:
        # Render header
        UIComponents.render_header()
        
        # Main application content
        with st.container():
            # Check if this is the first run
            if 'first_run' not in st.session_state:
                st.session_state.first_run = False
                show_welcome_message()
            
            # Run the main workflow
            run_summarization_workflow()
        
        # Render footer
        UIComponents.render_footer()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, check your inputs and API key.")

def show_welcome_message():
    """Show welcome message for first-time users."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="margin-bottom: 1rem;">👋 Welcome to Academic Paper Digest Generator!</h2>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Transform complex academic content into structured, digestible summaries using advanced AI.
        </p>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
            <div style="text-align: center;">
                <h3>📄 Multiple Sources</h3>
                <p>PDFs, YouTube, URLs</p>
            </div>
            <div style="text-align: center;">
                <h3>🎯 Smart Strategies</h3>
                <p>Stuff, Map-Reduce, Refine</p>
            </div>
            <div style="text-align: center;">
                <h3>🧠 Claude AI</h3>
                <p>Powered by Anthropic</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start instructions
    with st.expander("🚀 Quick Start Guide", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Step 1: Configure**
            1. Enter your Anthropic API key in the sidebar
            2. Select your preferred Claude model
            3. Adjust settings if needed
            """)
            
            st.markdown("""
            **Step 2: Input**
            1. Choose input type (PDF, YouTube, URL)
            2. Upload file or enter URL
            3. Verify content is loaded
            """)
        
        with col2:
            st.markdown("""
            **Step 3: Customize**
            1. Select summarization strategy
            2. Choose summary type
            3. Review recommendations
            """)
            
            st.markdown("""
            **Step 4: Generate**
            1. Click "Generate Academic Summary"
            2. Wait for processing
            3. Download your summary!
            """)

def setup_error_handling():
    """Setup global error handling for the application."""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        st.error("An unexpected error occurred. Please check the logs and try again.")
    
    import sys
    sys.excepthook = handle_exception

if __name__ == "__main__":
    # Setup error handling
    setup_error_handling()
    
    # Run the main application
    main()