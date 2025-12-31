"""
Enhanced Streamlit Database Chatbot Application
Includes user-friendly database configuration interface
"""
import streamlit as st
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from agents.sql_agent import get_sql_agent
from database.config import db_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Database Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'chat_history': [],
        'db_connected': False,
        'agent_initialized': False,
        'model_provider': "openai",
        'model_name': "gpt-4",
        'show_db_config': False,
        'connection_tested': False,
        'db_config_saved': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_database_configuration():
    """Show database configuration interface"""
    st.subheader("🔧 Database Configuration")
    
    with st.expander("Database Connection Settings", expanded=not st.session_state.db_connected):
        col1, col2 = st.columns(2)
        
        with col1:
            # Get current values from environment or use defaults
            current_host = os.getenv('MYSQL_HOST', 'localhost')
            current_port = int(os.getenv('MYSQL_PORT', 3306))
            current_user = os.getenv('MYSQL_USER', 'root')
            current_database = os.getenv('MYSQL_DATABASE', 'chatbot_db')
            
            db_host = st.text_input("MySQL Host", value=current_host, help="Database server hostname or IP address")
            db_port = st.number_input("MySQL Port", value=current_port, min_value=1, max_value=65535, help="Database server port (usually 3306)")
            db_user = st.text_input("MySQL Username", value=current_user, help="Database username")
            db_password = st.text_input("MySQL Password", type="password", help="Database password (supports special characters like @, #, %)")
        
        with col2:
            db_database = st.text_input("Database Name", value=current_database, help="Name of the database to connect to")
            
            # Connection info display
            st.markdown("### Current Connection Info")
            try:
                conn_info = db_config.get_connection_info()
                st.info(f"""
                **Host:** {conn_info['host']}:{conn_info['port']}  
                **User:** {conn_info['user']}  
                **Database:** {conn_info['database']}  
                **Password:** {'✅ Set' if conn_info['password_set'] else '❌ Not Set'} ({conn_info['password_length']} chars)
                """)
            except Exception as e:
                st.error(f"Error getting connection info: {e}")
        
        # Action buttons
        col_test, col_save, col_reset = st.columns(3)
        
        with col_test:
            if st.button("🔍 Test Connection", use_container_width=True, key="test_connection_config"):
                test_database_connection_with_params(db_host, db_port, db_user, db_password, db_database)
        
        with col_save:
            if st.button("💾 Save Configuration", use_container_width=True, key="save_configuration"):
                save_database_configuration(db_host, db_port, db_user, db_password, db_database)
        
        with col_reset:
            if st.button("🔄 Reset to Defaults", use_container_width=True, key="reset_configuration"):
                reset_database_configuration()

def test_database_connection_with_params(host, port, user, password, database):
    """Test database connection with provided parameters"""
    try:
        # Temporarily update environment variables for testing
        original_values = {}
        test_env_vars = {
            'MYSQL_HOST': host,
            'MYSQL_PORT': str(port),
            'MYSQL_USER': user,
            'MYSQL_PASSWORD': password,
            'MYSQL_DATABASE': database
        }
        
        # Store original values and set test values
        for key, value in test_env_vars.items():
            original_values[key] = os.getenv(key)
            os.environ[key] = value
        
        # Create a new database config instance for testing
        from database.config import DatabaseConfig
        test_db_config = DatabaseConfig()
        
        with st.spinner("Testing database connection..."):
            success = test_db_config.test_connection(detailed=True, debug=True)
            
        if success:
            st.success("✅ Database connection successful!")
            st.session_state.db_connected = True
            st.session_state.connection_tested = True
            
            # Show database info
            try:
                table_info = test_db_config.get_table_info()
                if table_info:
                    st.info(f"Found {len(table_info)} tables: {', '.join(table_info.keys())}")
                else:
                    st.warning("Database is empty (no tables found)")
            except Exception as e:
                st.warning(f"Connected but couldn't retrieve table info: {e}")
        else:
            st.error("❌ Database connection failed! Check the logs above for details.")
            st.session_state.db_connected = False
        
        # Restore original environment variables
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
                
    except Exception as e:
        st.error(f"❌ Connection test error: {e}")
        st.session_state.db_connected = False

def save_database_configuration(host, port, user, password, database):
    """Save database configuration to .env file"""
    try:
        env_content = f"""# Database Configuration
MYSQL_HOST={host}
MYSQL_PORT={port}
MYSQL_USER={user}
MYSQL_PASSWORD={password}
MYSQL_DATABASE={database}

# AI Model API Keys (at least one required)
OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')}
GROQ_API_KEY={os.getenv('GROQ_API_KEY', '')}

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_HEADLESS=true
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        # Update current environment
        os.environ['MYSQL_HOST'] = host
        os.environ['MYSQL_PORT'] = str(port)
        os.environ['MYSQL_USER'] = user
        os.environ['MYSQL_PASSWORD'] = password
        os.environ['MYSQL_DATABASE'] = database
        
        st.success("✅ Database configuration saved to .env file!")
        st.info("Please restart the application to load the new configuration.")
        st.session_state.db_config_saved = True
        
    except Exception as e:
        st.error(f"❌ Error saving configuration: {e}")

def reset_database_configuration():
    """Reset database configuration to defaults"""
    default_config = {
        'MYSQL_HOST': 'localhost',
        'MYSQL_PORT': '3306',
        'MYSQL_USER': 'root',
        'MYSQL_PASSWORD': '',
        'MYSQL_DATABASE': 'chatbot_db'
    }
    
    for key, value in default_config.items():
        os.environ[key] = value
    
    st.info("🔄 Configuration reset to defaults. Please update with your actual database credentials.")
    st.rerun()

def test_database_connection():
    """Test and display database connection status"""
    try:
        if db_config.test_connection(detailed=True):
            st.session_state.db_connected = True
            return True
        else:
            st.session_state.db_connected = False
            return False
    except Exception as e:
        st.session_state.db_connected = False
        logger.error(f"Database connection error: {e}")
        return False

def show_connection_status():
    """Show current connection status"""
    if st.session_state.db_connected:
        st.markdown('<div class="connection-status connected">🟢 Database Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection-status disconnected">🔴 Database Disconnected</div>', unsafe_allow_html=True)

def initialize_agent():
    """Initialize the SQL agent"""
    try:
        if not st.session_state.db_connected:
            st.error("Please establish database connection first")
            return None
        
        agent = get_sql_agent(
            model_provider=st.session_state.model_provider,
            model_name=st.session_state.model_name
        )
        st.session_state.agent_initialized = True
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.session_state.agent_initialized = False
        return None

def display_database_info():
    """Display database schema information"""
    if st.session_state.db_connected:
        try:
            table_info = db_config.get_table_info()
            
            if table_info:
                st.subheader("📊 Database Schema")
                
                for table_name, info in table_info.items():
                    with st.expander(f"📋 Table: {table_name} ({info['column_count']} columns)"):
                        # Display columns
                        col_df = pd.DataFrame(info['columns'])
                        st.dataframe(col_df, use_container_width=True)
                        
                        # Display sample data
                        sample_data = db_config.get_sample_data(table_name, limit=3)
                        if sample_data:
                            st.write("**Sample Data:**")
                            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
                        else:
                            st.info("No sample data available")
            else:
                st.warning("No tables found in the database")
                
        except Exception as e:
            st.error(f"Error loading database info: {str(e)}")

def format_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Format and display chat messages"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
        <strong>🧑 You:</strong><br>
        {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        message_class = "bot-message"
        if not message.get('success', True):
            message_class = "error-message"
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
        <strong>🤖 Assistant:</strong><br>
        {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display query results if available
        if 'results' in message and message['results']:
            try:
                df = pd.DataFrame(message['results'])
                st.dataframe(df, use_container_width=True)
                
                # Auto-generate simple visualizations for numeric data
                numeric_columns = df.select_dtypes(include=['number']).columns
                if len(numeric_columns) > 0 and len(df) > 1:
                    st.subheader("📈 Data Visualization")
                    
                    if len(numeric_columns) == 1:
                        # Single numeric column - histogram
                        fig = px.histogram(df, x=numeric_columns[0],
                                         title=f"Distribution of {numeric_columns[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(numeric_columns) >= 2:
                        # Multiple numeric columns - scatter plot
                        fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1],
                                       title=f"{numeric_columns[0]} vs {numeric_columns[1]}")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not display results as table: {str(e)}")

def process_user_question(question: str):
    """Process user question through the AI agent"""
    if not st.session_state.db_connected:
        st.error("Please connect to the database first")
        return
    
    if not st.session_state.agent_initialized:
        st.error("Please initialize the AI agent first")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': question,
        'timestamp': datetime.now()
    })
    
    # Process with agent
    try:
        with st.spinner("🤔 Thinking..."):
            agent = get_sql_agent()
            response = agent.chat(question)
        
        # Add bot response to chat history
        bot_message = {
            'type': 'bot',
            'content': response['answer'],
            'success': response['success'],
            'timestamp': datetime.now()
        }
        
        # Add results if available
        if 'intermediate_steps' in response and response['intermediate_steps']:
            # Try to extract SQL results from intermediate steps
            for step in response['intermediate_steps']:
                if len(step) > 1 and hasattr(step[1], 'tool_input'):
                    tool_input = step[1].tool_input
                    if isinstance(tool_input, dict) and 'query' in tool_input:
                        try:
                            query_result = db_config.execute_query(tool_input['query'])
                            if query_result:
                                bot_message['results'] = query_result
                                break
                        except:
                            pass
        
        st.session_state.chat_history.append(bot_message)
        
    except Exception as e:
        # Add error message to chat history
        st.session_state.chat_history.append({
            'type': 'bot',
            'content': f"Sorry, I encountered an error: {str(e)}",
            'success': False,
            'timestamp': datetime.now()
        })
    
    # Rerun to update the display
    st.rerun()

def show_example_questions():
    """Display example questions for users"""
    st.subheader("💡 Example Questions")
    
    examples = [
        "Show me all the tables in this database",
        "What are the column names in the first table?",
        "How many records are there in each table?",
        "Show me the first 10 rows from any table",
        "What is the structure of the database?",
        "Give me a summary of the data",
        "Show me some sample data from each table"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"📝 {example}", key=f"example_question_{i}"):
            process_user_question(example)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
    <h1>🤖 Database Chatbot</h1>
    <p>Interact with your MySQL database using natural language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database Configuration Section
        st.subheader("🗄️ Database")
        show_connection_status()
        
        if st.button("🔧 Configure Database", use_container_width=True, key="configure_db_sidebar"):
            st.session_state.show_db_config = True
        
        if st.button("🔍 Test Connection", use_container_width=True, key="test_connection_sidebar"):
            test_database_connection()
        
        # Model Configuration
        st.subheader("🤖 AI Model Settings")
        model_provider = st.selectbox(
            "Provider",
            ["openai", "groq"],
            index=0 if st.session_state.model_provider == "openai" else 1
        )
        
        if model_provider == "openai":
            model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        else:  # groq
            model_options = ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"]
        
        model_name = st.selectbox("Model", model_options, index=0)
        
        # Update session state if changed
        if model_provider != st.session_state.model_provider or model_name != st.session_state.model_name:
            st.session_state.model_provider = model_provider
            st.session_state.model_name = model_name
            st.session_state.agent_initialized = False
        
        # Initialize agent button
        if st.button("🚀 Initialize AI Agent", use_container_width=True, key="initialize_agent_sidebar"):
            with st.spinner("Initializing AI agent..."):
                agent = initialize_agent()
                if agent:
                    st.success("✅ Agent Ready")
                else:
                    st.error("❌ Agent Failed")
        
        # Display agent status
        if st.session_state.agent_initialized:
            st.success("🤖 Agent Ready")
        else:
            st.warning("🤖 Agent Not Ready")
        
        # Actions
        st.subheader("🎯 Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat_sidebar"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("📊 Show Database Schema", use_container_width=True, key="show_schema_sidebar"):
            st.session_state.show_schema = True
    
    # Main Content Area
    if st.session_state.get('show_db_config', False):
        show_database_configuration()
        if st.button("✅ Done with Configuration", key="done_configuration"):
            st.session_state.show_db_config = False
            st.rerun()
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Chat Interface")
            
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if message['type'] == 'user':
                    format_chat_message(message, is_user=True)
                else:
                    format_chat_message(message, is_user=False)
            
            # Chat input
            user_question = st.text_input(
                "Ask a question about your database:",
                placeholder="e.g., Show me all tables in the database",
                key="user_input"
            )
            
            col_send, col_example = st.columns([1, 1])
            
            with col_send:
                if st.button("📤 Send", type="primary", use_container_width=True, key="send_message"):
                    if user_question.strip():
                        process_user_question(user_question)
                    else:
                        st.warning("Please enter a question")
            
            with col_example:
                if st.button("💡 Show Examples", use_container_width=True, key="show_examples"):
                    show_example_questions()
            
            # Direct SQL Query Interface
            st.subheader("🔧 Direct SQL Query")
            with st.expander("Execute Custom SQL (Read-only)", expanded=False):
                sql_query = st.text_area(
                    "Enter your SQL query:",
                    placeholder="SELECT * FROM table_name LIMIT 10;",
                    height=100
                )
                
                if st.button("▶️ Execute SQL", key="execute_sql"):
                    if sql_query.strip():
                        execute_direct_sql(sql_query)
                    else:
                        st.warning("Please enter a SQL query")
        
        with col2:
            st.header("📊 Database Overview")
            
            # Connection status and info
            show_connection_status()
            
            if st.session_state.db_connected:
                try:
                    # Get database information
                    conn_info = db_config.get_connection_info()
                    
                    st.markdown("### 🔗 Connection Details")
                    st.info(f"""
                    **Host:** {conn_info['host']}:{conn_info['port']}  
                    **Database:** {conn_info['database']}  
                    **User:** {conn_info['user']}
                    """)
                    
                    # Show table summary
                    agent = get_sql_agent() if st.session_state.agent_initialized else None
                    if agent:
                        table_summary = agent.get_table_summary()
                        
                        st.metric("📋 Total Tables", table_summary.get("total_tables", 0))
                        
                        # Table information
                        st.subheader("📚 Tables")
                        for table_name, info in table_summary.get("tables", {}).items():
                            with st.expander(f"📋 {table_name}"):
                                st.write(f"**Columns:** {info['column_count']}")
                                st.write(f"**Has Data:** {'Yes' if info['has_data'] else 'No'}")
                                if info.get('columns'):
                                    st.write("**Column Names:**")
                                    for col in info['columns']:
                                        st.write(f"• {col}")
                    
                    # Quick stats
                    if st.button("📈 Generate Database Report", key="generate_report"):
                        generate_database_report()
                        
                except Exception as e:
                    st.error(f"Error loading database overview: {str(e)}")
            else:
                st.warning("Connect to database to see overview")
                st.markdown("""
                ### 🚀 Getting Started
                1. Configure your database connection
                2. Test the connection
                3. Initialize the AI agent
                4. Start asking questions!
                """)
    
    # Show database schema if requested
    if st.session_state.get('show_schema', False):
        display_database_info()
        st.session_state.show_schema = False

def execute_direct_sql(query: str):
    """Execute direct SQL query"""
    if not st.session_state.db_connected:
        st.error("Please connect to the database first")
        return
    
    try:
        # Safety check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        query_lower = query.lower().strip()
        
        if any(keyword in query_lower for keyword in dangerous_keywords):
            st.error("⚠️ Dangerous SQL operations are not allowed for safety reasons")
            return
        
        with st.spinner("Executing query..."):
            results = db_config.execute_query(query)
        
        if results:
            st.success(f"✅ Query executed successfully! Returned {len(results)} rows.")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Query executed but returned no results.")
            
    except Exception as e:
        st.error(f"❌ Error executing query: {str(e)}")

def generate_database_report():
    """Generate a comprehensive database report"""
    if not st.session_state.db_connected:
        st.error("Please connect to the database first")
        return
    
    try:
        with st.spinner("Generating database report..."):
            table_info = db_config.get_table_info()
            
            if not table_info:
                st.warning("No tables found in database")
                return
            
            st.subheader("📊 Database Report")
            
            # Summary statistics
            total_tables = len(table_info)
            total_columns = sum(info['column_count'] for info in table_info.values())
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📋 Total Tables", total_tables)
            col2.metric("📊 Total Columns", total_columns)
            col3.metric("📈 Avg Columns/Table", round(total_columns/total_tables, 1))
            
            # Table details
            report_data = []
            for table_name, info in table_info.items():
                sample_data = db_config.get_sample_data(table_name, limit=1)
                report_data.append({
                    'Table Name': table_name,
                    'Column Count': info['column_count'],
                    'Has Data': 'Yes' if sample_data else 'No',
                    'Sample Record Count': len(sample_data)
                })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(report_df, x='Table Name', y='Column Count', 
                        title='Column Count by Table')
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()