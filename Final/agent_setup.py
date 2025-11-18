"""
Setup module for initializing the reasoning agent and tools.
"""

import os
from tools import (
    ToolRegistry,
    SQLTool,
    RAGTool,
    PythonTool,
    DataGeneratorTool
)
from agents import ReActAgent


def create_tool_registry(db_config: dict, rag_collection, retrieve_func, format_func) -> ToolRegistry:
    """
    Create and register all tools.
    
    Args:
        db_config: Database configuration dict
        rag_collection: ChromaDB collection
        retrieve_func: RAG retrieval function
        format_func: RAG formatting function
        
    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()
    
    # Register SQL tool
    sql_tool = SQLTool(db_config)
    registry.register_tool(sql_tool)
    
    # Register RAG tool
    rag_tool = RAGTool(rag_collection, retrieve_func, format_func)
    registry.register_tool(rag_tool)
    
    # Register Python computation tool
    python_tool = PythonTool()
    registry.register_tool(python_tool)
    
    # Register data generator tool (needs SQL tool)
    data_gen_tool = DataGeneratorTool(sql_tool)
    registry.register_tool(data_gen_tool)
    
    return registry


def create_react_agent(
    model,
    tool_registry: ToolRegistry,
    schema: str,
    max_iterations: int = 10,
    verbose: bool = False
) -> ReActAgent:
    """
    Create a ReAct agent instance.
    
    Args:
        model: Gemini model instance
        tool_registry: ToolRegistry with registered tools
        schema: Database schema string
        max_iterations: Maximum ReAct iterations
        verbose: Whether to log reasoning steps
        
    Returns:
        ReActAgent instance
    """
    return ReActAgent(
        model=model,
        tool_registry=tool_registry,
        schema=schema,
        max_iterations=max_iterations,
        verbose=verbose
    )

