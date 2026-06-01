"""
Utility functions and helpers for the database chatbot
"""
import re
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

def sanitize_sql_query(query: str) -> str:
    """
    Sanitize SQL query to prevent SQL injection
    """
    # Remove comments
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    return query.strip()

def is_safe_query(query: str) -> bool:
    """
    Check if SQL query is safe (read-only operations)
    """
    query_lower = query.lower().strip()
    
    # Dangerous keywords that modify data
    dangerous_keywords = [
        'drop', 'delete', 'truncate', 'alter', 'create', 
        'insert', 'update', 'replace', 'merge', 'grant', 
        'revoke', 'commit', 'rollback'
    ]
    
    # Check for dangerous keywords
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', query_lower):
            return False
    
    # Must start with SELECT (or WITH for CTEs)
    if not (query_lower.startswith('select') or query_lower.startswith('with')):
        return False
    
    return True

def format_query_result(results: List[Dict[str, Any]], max_rows: int = 100) -> Dict[str, Any]:
    """
    Format query results for display
    """
    if not results:
        return {
            "formatted_results": [],
            "row_count": 0,
            "columns": [],
            "truncated": False
        }
    
    # Limit results if too many
    truncated = len(results) > max_rows
    display_results = results[:max_rows]
    
    # Get column names
    columns = list(results[0].keys()) if results else []
    
    # Format datetime objects
    formatted_results = []
    for row in display_results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                formatted_row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, (int, float)) and abs(value) > 1000000:
                formatted_row[key] = f"{value:,}"
            else:
                formatted_row[key] = value
        formatted_results.append(formatted_row)
    
    return {
        "formatted_results": formatted_results,
        "row_count": len(results),
        "columns": columns,
        "truncated": truncated
    }

def generate_query_suggestions(table_info: Dict[str, Any]) -> List[str]:
    """
    Generate query suggestions based on table structure
    """
    suggestions = []
    
    if not table_info:
        return suggestions
    
    # Basic suggestions
    suggestions.extend([
        "Show me all table names in this database",
        "How many records are in each table?",
        "What are the data types of all columns?"
    ])
    
    # Table-specific suggestions
    for table_name, info in table_info.items():
        columns = [col['name'] for col in info.get('columns', [])]
        
        # Basic table queries
        suggestions.append(f"Show me all columns in the {table_name} table")
        suggestions.append(f"Show me the first 10 rows from {table_name}")
        
        # If there are common column names, suggest specific queries
        common_patterns = {
            'id': f"How many unique IDs are in {table_name}?",
            'name': f"Show me all unique names from {table_name}",
            'email': f"Find all email addresses in {table_name}",
            'date': f"Show me the date range in {table_name}",
            'created_at': f"Show me recent entries from {table_name}",
            'updated_at': f"When was {table_name} last updated?",
            'price': f"What's the average price in {table_name}?",
            'amount': f"What's the total amount in {table_name}?",
            'status': f"Show me all status values in {table_name}",
            'count': f"What's the sum of all counts in {table_name}?"
        }
        
        for col_name in columns:
            col_lower = col_name.lower()
            for pattern, suggestion in common_patterns.items():
                if pattern in col_lower:
                    suggestions.append(suggestion)
                    break
    
    return list(set(suggestions))  # Remove duplicates

def extract_table_names_from_query(query: str) -> List[str]:
    """
    Extract table names from SQL query
    """
    # Simple regex to find table names after FROM, JOIN, UPDATE, INSERT INTO
    pattern = r'\b(?:FROM|JOIN|UPDATE|INSERT\s+INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, query, re.IGNORECASE)
    return list(set(matches))

def validate_database_config(config: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate database configuration
    """
    required_fields = ['host', 'user', 'password', 'database']
    missing_fields = []
    
    for field in required_fields:
        if not config.get(field):
            missing_fields.append(field)
    
    if missing_fields:
        return {
            "valid": False,
            "errors": [f"Missing required field: {field}" for field in missing_fields]
        }
    
    # Validate port
    try:
        port = int(config.get('port', 3306))
        if port < 1 or port > 65535:
            return {
                "valid": False,
                "errors": ["Port must be between 1 and 65535"]
            }
    except ValueError:
        return {
            "valid": False,
            "errors": ["Port must be a valid integer"]
        }
    
    return {"valid": True, "errors": []}

def format_error_message(error: Exception) -> str:
    """
    Format error messages for user display
    """
    error_str = str(error)
    
    # Common database errors
    error_mappings = {
        "Access denied": "Database access denied. Please check your username and password.",
        "Unknown database": "Database not found. Please check the database name.",
        "Connection refused": "Cannot connect to database server. Please check the host and port.",
        "Table doesn't exist": "The specified table does not exist in the database.",
        "Column not found": "One or more columns in your query do not exist.",
        "Syntax error": "There's a syntax error in your SQL query.",
        "Timeout": "Database query timed out. Please try a simpler query."
    }
    
    for key, friendly_message in error_mappings.items():
        if key.lower() in error_str.lower():
            return friendly_message
    
    return f"Database error: {error_str}"

def generate_chart_config(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate chart configuration based on DataFrame structure
    """
    if df.empty:
        return None
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Suggest chart type based on data structure
    if len(numeric_cols) == 1 and len(categorical_cols) == 1:
        return {
            "type": "bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
        }
    elif len(numeric_cols) >= 2:
        return {
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}"
        }
    elif len(date_cols) == 1 and len(numeric_cols) >= 1:
        return {
            "type": "line",
            "x": date_cols[0],
            "y": numeric_cols[0],
            "title": f"{numeric_cols[0]} over time"
        }
    elif len(numeric_cols) == 1:
        return {
            "type": "histogram",
            "x": numeric_cols[0],
            "title": f"Distribution of {numeric_cols[0]}"
        }
    
    return None

def log_user_interaction(action: str, details: Dict[str, Any] = None):
    """
    Log user interactions for analytics
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details or {}
    }
    
    logger.info(f"User interaction: {json.dumps(log_entry)}")

class QueryCache:
    """Simple in-memory cache for query results"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hash(query.strip().lower())
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(query)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() - entry['timestamp'] > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            return None
        
        return entry['result']
    
    def set(self, query: str, result: Any):
        """Cache query result"""
        key = self._generate_key(query)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

# Global cache instance
query_cache = QueryCache()