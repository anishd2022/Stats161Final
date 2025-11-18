"""
Base class for all tools used by the reasoning agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        if self.success:
            return f"Tool executed successfully. Output: {self.output}"
        else:
            return f"Tool execution failed: {self.error}"


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with success status, output, and optional error
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing this tool's inputs and outputs.
        Used by the agent to understand how to call the tool.
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameter_schema()
        }
    
    @abstractmethod
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return schema for tool parameters.
        Should describe each parameter: name, type, description, required.
        """
        pass
    
    def validate_args(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate tool arguments before execution.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None

