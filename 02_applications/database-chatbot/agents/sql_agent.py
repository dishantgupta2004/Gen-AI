import os
import logging
from typing import Any, Dict, List, Optional
from langchain_openai import OpenAI, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage
from database.config import db_config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SQLChatAgent:
    def __init__(self, model_provider: str = 'openai', model_name: str='gpt-4', temperature: float = 0):
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.db = None
        self.agent_executor = None
        self._initialize_components()
        
    def _initialize_components(self):
        try:
            self._initialize_llm()
            self._initialize_database()
            self._create_agent()
            logger.info("SQLChatAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing SQLChatAgent: {e}")
            raise
        
    def _initialize_llm(self):
        try:
            if self.model_provider.lower() == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set.")
                
                self.llm = ChatOpenAI(
                    model= self.model_name,
                    temperature=self.temperature,
                    openai_api_key=api_key,
                    streaming=True
                )
                
            
            elif self.model_provider.lower() == 'groq':
                api_key = os.getenv('GROQ_API_KEY')
                if not api_key:
                    raise ValueError("GROQ_API_KEY environment variable is not set.")
                
                self.llm = ChatGroq(
                    model= self.model_name,
                    temperature=self.temperature,
                    api_key= api_key
                )
                
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
            
            logger.info(f"LLM initialized with model: {self.model_name} from provider: {self.model_provider}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
        
    
    def _initialize_database(self):
        """Initialize database connection for LangChain"""
        try:
            if not db_config.test_connection():
                raise Exception("Database connection test failed")
            
            connection_url = db_config.get_connection_url()
            self.db = SQLDatabase.from_uri(connection_url)
            
            logger.info("Database connection initialized for LangChain")
            
        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            raise
        
        
    
    def _create_agent(self):
        """Create SQL agent with enhanced capabilities"""
        try:
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Create agent with custom prompt
            self.agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
            logger.info("SQL Agent created successfully")
            
        except Exception as e:
            logger.error(f"Error creating SQL agent: {e}")
            raise
        
    def get_database_schema(self) -> str:
        try:
            return self.db.get_table_info()
        except Exception as e:
            logger.error(f"error getting database schema: {e}")
            return str(e)
        
    def chat(self, question: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing question: {question}")
            
            enhanced_question= f"""
            You are an expert SQL Analyst. Please help me with the following question about our database:
            Question: {question}
            
            Guidelines:
            1. First understand what the user is asking for.
            2. Examine the database schema to identify relevant tables and columns.
            3. Write clean, efficient SQL Queries.
            4. Provide clear explanations of your approach.
            5. If the question is ambiguous, ask clarifying questions.
            6. Always double-check your SQL Syntax.
            7. Provide meaningful insights from the results.
            
            Please be thorough in your response and explain your reasoning.
            """
            response = self.agent_executor.invoke(
                {"input": enhanced_question}
            )
            
            result = {
                'question': question,
                'answer': response.get('output', 'No response generated'),
                'success': True,
                'intermediate_steps': response.get('intermediate_steps', [])
            }
            
            logger.info(f"Response generated successfully: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return {
                'question': question,
                'answer': str(e),
                'success': False,
                'intermediate_steps': []
            }
            
    def get_table_summary(self) -> Dict[str, Any]:
        try:
            table_info = db_config.get_table_info()
            summary = {
                'total_tables': len(table_info),
                'tables': {}
            }
            
            for table_name, info in table_info.items():
                sample_data = db_config.get_sample_data(table_name, limit= 2)
                summary['tables'][table_name] = {
                    'column_count': info['column_count'],
                    'columns': [col['name'] for col in info['columns']],
                    'sample_data_count': len(sample_data),
                    'has_data': len(sample_data) > 0
                }
                
            return summary
        
        except Exception as e:
            logger.error(f"Error getting table summary: {e}")
            return {
                'error': str(e),
                'success': False
            }
            
            
    def execute_direct_query(self, query: str) -> Dict[str, Any]:
        try:
            query_lower = query.lower().strip()
            
            dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
            if any(keyword in query_lower for keyword in dangerous_keywords):
                return {
                    "success": False,
                    "error": "Dangerous SQL operations are not allowed through direct query execution",
                    "query": query
                }
                
            results= db_config.execute_query(query)
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'row_count': len(results),
            }
            
        except Exception as e:
            logger.error(f"Error executing direct query: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
            
            
            
## global agent instance
sql_agent = None

def get_sql_agent(model_provider: str = 'openai', model_name: str='gpt-4') -> SQLChatAgent:
    global sql_agent
    if sql_agent is None:
        sql_agent = SQLChatAgent(model_provider, model_name)
    return sql_agent