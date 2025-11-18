"""
Tool Registry - Manages all available tools for the reasoning agent.
"""

from typing import Dict, List, Optional
from .base_tool import BaseTool


class ToolRegistry:
    """Registry for managing and accessing tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def format_tools_for_llm(self) -> str:
        """
        Format all tool schemas as a string for LLM prompt.
        This helps the agent understand what tools are available.
        """
        schemas = self.get_tool_schemas()
        
        formatted = "Available Tools:\n"
        formatted += "=" * 80 + "\n\n"
        
        for schema in schemas:
            formatted += f"Tool: {schema['name']}\n"
            formatted += f"Description: {schema['description']}\n"
            formatted += "Parameters:\n"
            
            for param_name, param_info in schema['parameters'].items():
                required = param_info.get('required', False)
                param_type = param_info.get('type', 'any')
                description = param_info.get('description', '')
                default = param_info.get('default', None)
                
                req_marker = " (required)" if required else " (optional)"
                default_str = f" [default: {default}]" if default is not None else ""
                
                formatted += f"  - {param_name} ({param_type}){req_marker}{default_str}: {description}\n"
            
            formatted += "\n"
        
        return formatted

