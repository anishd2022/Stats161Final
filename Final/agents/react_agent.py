"""
ReAct Agent - Reasoning and Acting agent using the ReAct pattern.
Implements: Thought → Action → Observation → Repeat
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from tools.tool_registry import ToolRegistry
from tools.base_tool import ToolResult


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    thought: str
    action: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    tool_result: Optional[ToolResult] = None


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.
    
    The agent follows this pattern:
    1. Thought: Reason about what to do next
    2. Action: Decide which tool to use and with what arguments
    3. Observation: Execute tool and observe results
    4. Repeat until final answer is ready
    """
    
    def __init__(
        self,
        model,
        tool_registry: ToolRegistry,
        schema: str,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        """
        Initialize ReAct agent.
        
        Args:
            model: Gemini model instance
            tool_registry: Registry of available tools
            schema: Database schema string
            max_iterations: Maximum number of ReAct iterations
            verbose: Whether to log reasoning steps
        """
        self.model = model
        self.tool_registry = tool_registry
        self.schema = schema
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.steps: List[ReActStep] = []
    
    def run(self, user_query: str, rag_context: str = "") -> Dict[str, Any]:
        """
        Run the ReAct agent on a user query.
        
        Args:
            user_query: User's question
            rag_context: Optional RAG context from documentation
            
        Returns:
            Dictionary with final_answer, steps, and metadata
        """
        self.steps = []
        
        # Start a chat session for this query
        chat_session = self.model.start_chat(history=[])
        
        # Build and send initial system prompt
        system_prompt = self._build_system_prompt(rag_context)
        initial_message = f"{system_prompt}\n\nUser Question: {user_query}\n\nWhat should I do first? Think step by step, then decide on an action."
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- ReAct Iteration {iteration + 1} ---")
            
            # Step 1: Thought - Agent reasons about what to do
            try:
                if iteration == 0:
                    # First iteration: send initial message
                    thought_response = chat_session.send_message(initial_message)
                else:
                    # Subsequent iterations: ask what to do next
                    thought_prompt = "What should I do next? Think step by step, then decide on an action. Format: Thought: [reasoning] Action: tool_name(args) or FINAL_ANSWER: [answer]"
                    thought_response = chat_session.send_message(thought_prompt)
                
                thought_text = thought_response.text.strip()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error getting agent response: {str(e)}",
                    "steps": self.steps,
                    "iterations": iteration + 1
                }
            
            if self.verbose:
                print(f"Thought: {thought_text[:200]}...")
            
            # Check if agent thinks it's done
            if self._is_final_answer(thought_text):
                final_answer = self._extract_final_answer(thought_text)
                if not final_answer:
                    # If extraction failed, use the text as-is
                    final_answer = thought_text
                return {
                    "success": True,
                    "answer": final_answer,
                    "steps": self.steps,
                    "iterations": iteration + 1,
                    "reasoning_visible": False  # Hide from user by default
                }
            
            # Step 2: Action - Agent decides which tool to use
            action = self._parse_action(thought_text)
            
            if not action:
                # Agent couldn't parse action, try to extract final answer
                final_answer = self._extract_final_answer(thought_text)
                if final_answer:
                    return {
                        "success": True,
                        "answer": final_answer,
                        "steps": self.steps,
                        "iterations": iteration + 1,
                        "reasoning_visible": False
                    }
                else:
                    # Error - agent is stuck, ask it to try again
                    error_msg = "I couldn't parse your action. Please format it as: Action: tool_name(arg1=value1, arg2=value2) or provide FINAL_ANSWER: [your answer]"
                    chat_session.send_message(error_msg)
                    continue
            
            step = ReActStep(thought=thought_text, action=action)
            self.steps.append(step)
            
            if self.verbose:
                print(f"Action: {action['tool']} with args: {action.get('args', {})}")
            
            # Step 3: Observation - Execute tool and get results
            tool_result = self._execute_action(action)
            observation = self._format_observation(tool_result)
            
            step.observation = observation
            step.tool_result = tool_result
            
            if self.verbose:
                print(f"Observation: {observation[:200]}...")
            
            # Format observation for chat (truncate if too long)
            observation_for_chat = observation
            if len(observation_for_chat) > 2000:
                observation_for_chat = observation_for_chat[:2000] + "\n[Results truncated for length...]"
            
            # Send observation back to agent
            observation_message = f"Observation from {action['tool']}:\n{observation_for_chat}"
            chat_session.send_message(observation_message)
            
            # Check if we have enough information to answer
            if tool_result.success and self._has_enough_information():
                # Ask agent to finalize answer
                final_prompt = f"Based on all the information gathered, provide a clear, concise answer to the user's original question: '{user_query}'. Format your response as: FINAL_ANSWER: [your answer]"
                try:
                    final_response = chat_session.send_message(final_prompt)
                    final_answer = final_response.text.strip()
                    
                    # Extract final answer if it has the marker
                    if "FINAL_ANSWER:" in final_answer.upper():
                        final_answer = self._extract_final_answer(final_answer) or final_answer
                    
                    return {
                        "success": True,
                        "answer": final_answer,
                        "steps": self.steps,
                        "iterations": iteration + 1,
                        "reasoning_visible": False
                    }
                except Exception as e:
                    # If finalization fails, use the last successful result
                    return {
                        "success": True,
                        "answer": f"Based on the data retrieved: {observation[:500]}",
                        "steps": self.steps,
                        "iterations": iteration + 1,
                        "reasoning_visible": False
                    }
        
        # Max iterations reached - ask for final answer anyway
        try:
            final_prompt = f"Maximum iterations reached. Please provide a final answer to: '{user_query}'. Format: FINAL_ANSWER: [your answer]"
            final_response = chat_session.send_message(final_prompt)
            final_answer = final_response.text.strip()
            if "FINAL_ANSWER:" in final_answer.upper():
                final_answer = self._extract_final_answer(final_answer) or final_answer
            
            return {
                "success": True,
                "answer": final_answer,
                "steps": self.steps,
                "iterations": self.max_iterations,
                "reasoning_visible": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Maximum iterations ({self.max_iterations}) reached without final answer. Last error: {str(e)}",
                "steps": self.steps,
                "iterations": self.max_iterations
            }
    
    def _build_system_prompt(self, rag_context: str = "") -> str:
        """Build the system prompt with tool information."""
        tools_info = self.tool_registry.format_tools_for_llm()
        
        rag_section = ""
        if rag_context:
            rag_section = f"\n\nAdditional Context from Documentation:\n{rag_context}\n"
        
        prompt = f"""You are a CTR Reasoning Agent. Your job is to answer questions about the CTR dataset by reasoning through the problem and using available tools.

Database Schema:
{self.schema}
{rag_section}

{tools_info}

ReAct Pattern Instructions:
1. **Thought**: Think about what you need to do to answer the question. Break down the problem into steps.
2. **Action**: Decide which tool to use. Format your action as: "Action: tool_name(arg1=value1, arg2=value2)"
3. **Observation**: After the tool executes, you'll see the results. Use this to inform your next thought.
4. **Repeat**: Continue until you have enough information to provide a final answer.

When you have enough information, provide your final answer starting with "FINAL_ANSWER:"

Important:
- Use tools to gather data, compute metrics, or retrieve documentation
- For SQL queries, use the run_sql tool
- For CTR calculations, use the compute_python tool
- For documentation, use the retrieve_docs tool
- For generating recommendations, use the generate_data tool
- Always think step-by-step before taking action
- If a tool fails, reason about why and try a different approach

Let's begin."""
        
        return prompt
    
    
    def _parse_action(self, thought_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse action from thought text.
        Looks for: "Action: tool_name(arg1=value1, arg2=value2)"
        """
        # Try to find Action: pattern
        action_match = re.search(r'Action:\s*(\w+)\s*\(([^)]*)\)', thought_text, re.IGNORECASE)
        if not action_match:
            # Try alternative format
            action_match = re.search(r'action:\s*(\w+)\s*\(([^)]*)\)', thought_text, re.IGNORECASE)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1).strip()
        args_str = action_match.group(2).strip()
        
        # Parse arguments
        args = {}
        if args_str:
            # Simple parsing - split by comma and parse key=value pairs
            arg_parts = [a.strip() for a in args_str.split(',')]
            for arg_part in arg_parts:
                if '=' in arg_part:
                    key, value = arg_part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse value types
                    if value.startswith('"') and value.endswith('"'):
                        args[key] = value[1:-1]  # Remove quotes
                    elif value.startswith("'") and value.endswith("'"):
                        args[key] = value[1:-1]
                    elif value.lower() in ['true', 'false']:
                        args[key] = value.lower() == 'true'
                    elif value.isdigit():
                        args[key] = int(value)
                    elif self._is_float(value):
                        args[key] = float(value)
                    else:
                        args[key] = value
        
        return {
            "tool": tool_name,
            "args": args
        }
    
    def _is_float(self, s: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _execute_action(self, action: Dict[str, Any]) -> ToolResult:
        """Execute a tool action."""
        tool_name = action.get("tool")
        args = action.get("args", {})
        
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_registry.list_tools())}"
            )
        
        try:
            return tool.execute(**args)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error executing tool {tool_name}: {str(e)}"
            )
    
    def _format_observation(self, tool_result: ToolResult) -> str:
        """Format tool result as observation text."""
        if not tool_result.success:
            return f"Error: {tool_result.error}"
        
        # Format based on tool result type
        if isinstance(tool_result.output, str):
            return tool_result.output
        elif hasattr(tool_result.output, 'to_string'):  # DataFrame
            return tool_result.output.to_string(index=False)
        else:
            return str(tool_result.output)
    
    def _format_action_for_llm(self, action: Dict[str, Any]) -> str:
        """Format action for LLM conversation history."""
        tool_name = action.get("tool")
        args = action.get("args", {})
        args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        return f"{tool_name}({args_str})"
    
    def _is_final_answer(self, text: str) -> bool:
        """Check if text contains a final answer."""
        return "FINAL_ANSWER:" in text.upper() or text.strip().startswith("FINAL_ANSWER:")
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from text."""
        # Look for FINAL_ANSWER: marker
        match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no marker, return the text itself if it looks like an answer
        if len(text) > 50 and not "Action:" in text:
            return text.strip()
        
        return None
    
    def _has_enough_information(self) -> bool:
        """Heuristic to check if we have enough information to answer."""
        # Simple heuristic: if we've executed at least one successful tool
        # and the last observation is not an error
        if len(self.steps) == 0:
            return False
        
        last_step = self.steps[-1]
        if last_step.tool_result and last_step.tool_result.success:
            # Check if we've gathered some data
            return True
        
        return False

