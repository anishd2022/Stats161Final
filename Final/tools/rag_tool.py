"""
RAG Tool - Retrieves relevant documentation using the RAG system.
"""

from typing import Dict, Any, Optional, List
from .base_tool import BaseTool, ToolResult


class RAGTool(BaseTool):
    """Tool for retrieving relevant documentation chunks."""
    
    def __init__(self, rag_collection, retrieve_func, format_func):
        """
        Initialize RAG tool.
        
        Args:
            rag_collection: ChromaDB collection object
            retrieve_func: Function to retrieve context (from rag_system)
            format_func: Function to format context (from rag_system)
        """
        super().__init__(
            name="retrieve_docs",
            description="Retrieve relevant documentation about the CTR dataset, schema, variables, and domain knowledge."
        )
        self.rag_collection = rag_collection
        self.retrieve_func = retrieve_func
        self.format_func = format_func
    
    def execute(self, query: str, top_k: int = 3) -> ToolResult:
        """
        Retrieve relevant documentation.
        
        Args:
            query: Search query
            top_k: Number of top results to retrieve
            
        Returns:
            ToolResult with formatted context string
        """
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="Query cannot be empty"
            )
        
        if not self.rag_collection:
            return ToolResult(
                success=False,
                output=None,
                error="RAG collection not available. Documentation retrieval is disabled."
            )
        
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieve_func(query, self.rag_collection, top_k=top_k)
            
            if not retrieved_chunks:
                return ToolResult(
                    success=True,
                    output="No relevant documentation found for this query.",
                    metadata={"chunks_retrieved": 0}
                )
            
            # Format context
            formatted_context = self.format_func(retrieved_chunks)
            
            return ToolResult(
                success=True,
                output=formatted_context,
                metadata={
                    "chunks_retrieved": len(retrieved_chunks),
                    "sources": [chunk.get('source', 'unknown') for chunk in retrieved_chunks]
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"RAG retrieval error: {str(e)}"
            )
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Search query to find relevant documentation",
                "required": True
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to retrieve (default: 3)",
                "required": False,
                "default": 3
            }
        }

