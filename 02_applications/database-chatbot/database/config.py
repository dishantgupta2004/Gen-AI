import os
import logging
import time
from typing import Optional, Dict, Any
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    def __init__(self):
        # Load database configuration from environment with defaults
        self.host = os.environ.get('MYSQL_HOST', 'localhost')
        self.port = int(os.environ.get('MYSQL_PORT', 3306))
        self.user = os.environ.get('MYSQL_USER')
        self.password = os.environ.get('MYSQL_PASSWORD')
        self.database = os.environ.get('MYSQL_DATABASE')
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        
        # Validate required parameters
        self._validate_config()
    
    def _validate_config(self):
        """Validate database configuration parameters"""
        missing_params = []
        
        if not self.user:
            missing_params.append('MYSQL_USER')
        if not self.password:
            missing_params.append('MYSQL_PASSWORD')
        if not self.database:
            missing_params.append('MYSQL_DATABASE')
            
        if missing_params:
            error_msg = f"Missing required database configuration: {', '.join(missing_params)}"
            logger.error(error_msg)
            logger.error("Please check your .env file and ensure all required variables are set")
            raise ValueError(error_msg)
    
    def get_connection_url(self, include_db: bool = True, debug: bool = False) -> str:
        """Generate database connection URL with proper password encoding"""
        # Encode password to handle special characters like @, #, %, etc.
        encoded_password = quote_plus(self.password)
        
        if debug:
            logger.info(f"[DEBUG] Original password length: {len(self.password)}")
            logger.info(f"[DEBUG] Encoded password: {encoded_password}")
            logger.info(f"[DEBUG] Connection details: {self.user}@{self.host}:{self.port}")
        
        base_url = f"mysql+mysqlconnector://{self.user}:{encoded_password}@{self.host}:{self.port}"
        if include_db:
            return f"{base_url}/{self.database}"
        return base_url
    
    def create_engine(self, retry_count: int = 3, debug: bool = False) -> Engine:
        """Create database engine with retry logic and better error handling"""
        if self.engine:
            return self.engine
            
        for attempt in range(retry_count):
            try:
                connection_url = self.get_connection_url(debug=debug)
                logger.info(f"Attempting to create database engine (attempt {attempt + 1}/{retry_count})")
                logger.info(f"Connecting to: {self.host}:{self.port}/{self.database} as {self.user}")
                
                self.engine = create_engine(
                    connection_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    pool_timeout=30,
                    max_overflow=10,
                    echo=debug,  # Enable SQL echo for debugging
                    connect_args={
                        'connect_timeout': 60,
                        'autocommit': True,
                        'charset': 'utf8mb4',
                        'use_unicode': True
                    }
                )
                
                # Test the connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    
                logger.info(f"Database engine created successfully for {self.database} at {self.host}:{self.port}")
                return self.engine
                
            except OperationalError as e:
                error_code = getattr(e.orig, 'errno', None) if hasattr(e, 'orig') else None
                logger.error(f"Database connection attempt {attempt + 1} failed with error code {error_code}: {e}")
                
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._diagnose_connection_error(e)
                    raise ConnectionError(f"Failed to connect to database after {retry_count} attempts. Last error: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error creating database engine: {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(2)
    
    def _diagnose_connection_error(self, error: Exception):
        """Provide detailed diagnosis of connection errors"""
        error_msg = str(error).lower()
        
        logger.error("❌ Database Connection Failed - Diagnosis:")
        
        if "can't connect to mysql server" in error_msg:
            logger.error("   • MySQL server is not running or not accessible")
            logger.error(f"   • Check if MySQL is running on {self.host}:{self.port}")
            logger.error("   • Verify firewall settings and network connectivity")
            
        elif "access denied" in error_msg:
            logger.error("   • Authentication failed")
            logger.error(f"   • Check username: {self.user}")
            logger.error("   • Check password (verify special characters are handled)")
            logger.error("   • Verify user has necessary privileges")
            
        elif "unknown database" in error_msg:
            logger.error(f"   • Database '{self.database}' does not exist")
            logger.error("   • Create the database or check the database name")
            
        elif "connection refused" in error_msg:
            logger.error("   • Connection refused by server")
            logger.error(f"   • MySQL server might not be running on {self.host}:{self.port}")
            logger.error("   • Check if the port is correct")
            
        elif "timeout" in error_msg:
            logger.error("   • Connection timeout")
            logger.error("   • Server might be overloaded or network is slow")
            logger.error("   • Try increasing connection timeout")
            
        else:
            logger.error(f"   • Unexpected error: {error}")
    
    def test_connection(self, detailed: bool = False, debug: bool = False) -> bool:
        """Test database connection with detailed diagnostics"""
        try:
            logger.info("Testing database connection...")
            
            # Step 1: Check if we can create an engine
            engine = self.create_engine(debug=debug)
            
            # Step 2: Test basic connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                if test_value != 1:
                    logger.error("Basic SELECT test failed")
                    return False
            
            # Step 3: Test database access
            with engine.connect() as conn:
                conn.execute(text(f"USE `{self.database}`"))
                
            # Step 4: Test table listing (if detailed)
            if detailed:
                with engine.connect() as conn:
                    result = conn.execute(text("SHOW TABLES"))
                    tables = result.fetchall()
                    logger.info(f"Found {len(tables)} tables in database")
                    if tables:
                        logger.info("Tables: " + ", ".join([table[0] for table in tables]))
            
            logger.info("✅ Database connection test successful")
            return True
            
        except Exception as e:
            self._diagnose_connection_error(e)
            return False
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get table information with better error handling"""
        try:
            engine = self.create_engine()
            self.metadata = MetaData()
            
            # Use reflection to get table info
            with engine.connect() as conn:
                self.metadata.reflect(bind=conn)
            
            table_info = {}
            for table_name in self.metadata.tables:
                table = self.metadata.tables[table_name]
                columns = []
                
                for column in table.columns:
                    columns.append({
                        'name': column.name,
                        'type': str(column.type),
                        'nullable': column.nullable,
                        'primary_key': column.primary_key,
                        'default': str(column.default) if column.default else None
                    })
                
                table_info[table_name] = {
                    'columns': columns,
                    'column_count': len(columns)
                }
            
            logger.info(f"Successfully retrieved information for {len(table_info)} tables")
            return table_info
            
        except Exception as e:
            logger.error(f"Error retrieving table information: {e}")
            return {}
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> list:
        """Get sample data from a specific table"""
        try:
            engine = self.create_engine()
            with engine.connect() as conn:
                # Safely quote table name to prevent SQL injection
                quoted_table = f"`{table_name}`"
                query = text(f"SELECT * FROM {quoted_table} LIMIT :limit")
                result = conn.execute(query, {"limit": limit})
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return []
    
    def execute_query(self, query: str) -> list:
        """Execute a SQL query and return results"""
        try:
            engine = self.create_engine()
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query was: {query}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for display"""
        return {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'database': self.database,
            'password_set': bool(self.password),
            'password_length': len(self.password) if self.password else 0
        }

# Create global instance
db_config = DatabaseConfig()