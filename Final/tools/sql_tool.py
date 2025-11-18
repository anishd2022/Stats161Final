"""
SQL Tool - Executes SQL queries on the CTR database.
"""

import pandas as pd
import pymysql
from typing import Dict, Any, Optional
from .base_tool import BaseTool, ToolResult


class SQLTool(BaseTool):
    """Tool for executing SQL queries on the database."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize SQL tool with database configuration.
        
        Args:
            db_config: Dictionary with keys: host, user, password, database, port
        """
        super().__init__(
            name="run_sql",
            description="Execute SQL queries on the CTR database. Returns query results as a table or row count."
        )
        self.db_config = db_config
    
    def _get_connection(self):
        """Create a new database connection."""
        return pymysql.connect(
            host=self.db_config['host'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['database'],
            port=self.db_config.get('port', 3306),
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def execute(self, query: str, max_rows: int = 1000) -> ToolResult:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            max_rows: Maximum number of rows to return (for SELECT queries)
            
        Returns:
            ToolResult with DataFrame (for SELECT) or row count (for other queries)
        """
        # Validate query
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="Query cannot be empty"
            )
        
        # Basic safety check - prevent destructive operations
        query_upper = query.strip().upper()
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE TABLE', 'CREATE DATABASE']
        if any(keyword in query_upper for keyword in dangerous_keywords):
            return ToolResult(
                success=False,
                output=None,
                error=f"Query contains potentially dangerous operation. Blocked keywords: {dangerous_keywords}"
            )
        
        connection = None
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query_upper.startswith('SELECT'):
                results = cursor.fetchall()
                if results:
                    df = pd.DataFrame(results)
                    
                    # Truncate if too many rows
                    if len(df) > max_rows:
                        df_truncated = df.head(max_rows)
                        metadata = {
                            "total_rows": len(df),
                            "returned_rows": max_rows,
                            "truncated": True
                        }
                        return ToolResult(
                            success=True,
                            output=df_truncated,
                            metadata=metadata
                        )
                    else:
                        return ToolResult(
                            success=True,
                            output=df,
                            metadata={"total_rows": len(df), "truncated": False}
                        )
                else:
                    return ToolResult(
                        success=True,
                        output=pd.DataFrame(),
                        metadata={"total_rows": 0, "message": "Query returned no rows"}
                    )
            else:
                # For non-SELECT queries, commit and return row count
                connection.commit()
                rowcount = cursor.rowcount
                return ToolResult(
                    success=True,
                    output=rowcount,
                    metadata={"rowcount": rowcount, "message": f"Query executed successfully. {rowcount} row(s) affected."}
                )
                
        except Exception as e:
            if connection:
                connection.rollback()
            return ToolResult(
                success=False,
                output=None,
                error=f"SQL execution error: {str(e)}"
            )
        finally:
            if connection:
                connection.close()
            if 'cursor' in locals():
                cursor.close()
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "SQL query to execute (MySQL syntax)",
                "required": True
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum number of rows to return (default: 1000)",
                "required": False,
                "default": 1000
            }
        }
    
    def format_result_for_llm(self, result: ToolResult) -> str:
        """
        Format tool result as text for LLM consumption.
        
        Args:
            result: ToolResult from execute()
            
        Returns:
            Formatted string representation
        """
        if not result.success:
            return f"Error: {result.error}"
        
        if isinstance(result.output, pd.DataFrame):
            if len(result.output) == 0:
                return "Query returned no rows."
            
            # Format as text table
            text_output = result.output.to_string(index=False)
            
            if result.metadata and result.metadata.get("truncated"):
                text_output += f"\n\n[Note: Showing first {result.metadata['returned_rows']} of {result.metadata['total_rows']} total rows]"
            
            return text_output
        elif isinstance(result.output, int):
            return f"Query executed successfully. {result.output} row(s) affected."
        else:
            return str(result.output)

