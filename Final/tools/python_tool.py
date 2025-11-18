"""
Python Tool - Executes Python computations for CTR analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_tool import BaseTool, ToolResult


class PythonTool(BaseTool):
    """Tool for executing Python computations (CTR calculations, correlations, etc.)."""
    
    def __init__(self):
        super().__init__(
            name="compute_python",
            description="Execute Python computations for CTR analysis: calculate CTR metrics, correlations, aggregations, statistical tests, and data transformations."
        )
    
    def execute(self, operation: str, data: Any = None, **kwargs) -> ToolResult:
        """
        Execute a Python computation.
        
        Args:
            operation: Type of operation to perform (e.g., 'calculate_ctr', 'correlation', 'aggregate')
            data: Input data (DataFrame, dict, or other)
            **kwargs: Additional operation-specific parameters
            
        Returns:
            ToolResult with computation results
        """
        if not operation:
            return ToolResult(
                success=False,
                output=None,
                error="Operation type is required"
            )
        
        try:
            operation_lower = operation.lower()
            
            if operation_lower == "calculate_ctr":
                return self._calculate_ctr(data, **kwargs)
            elif operation_lower == "correlation":
                return self._calculate_correlation(data, **kwargs)
            elif operation_lower == "aggregate":
                return self._aggregate(data, **kwargs)
            elif operation_lower == "percentage_change":
                return self._percentage_change(data, **kwargs)
            elif operation_lower == "statistical_summary":
                return self._statistical_summary(data, **kwargs)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}. Supported operations: calculate_ctr, correlation, aggregate, percentage_change, statistical_summary"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Python computation error: {str(e)}"
            )
    
    def _calculate_ctr(self, data: Any, clicks_col: str = "clicks", impressions_col: str = "impressions") -> ToolResult:
        """Calculate CTR (Click-Through Rate) from data."""
        if data is None:
            return ToolResult(
                success=False,
                output=None,
                error="Data is required for CTR calculation"
            )
        
        if isinstance(data, pd.DataFrame):
            if clicks_col not in data.columns or impressions_col not in data.columns:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Required columns not found. Need: {clicks_col}, {impressions_col}"
                )
            
            # Calculate CTR
            data = data.copy()
            data['ctr'] = (data[clicks_col] / data[impressions_col] * 100).round(4)
            data['ctr'] = data['ctr'].replace([np.inf, -np.inf], np.nan)
            
            return ToolResult(
                success=True,
                output=data,
                metadata={"operation": "calculate_ctr", "ctr_column": "ctr"}
            )
        else:
            # For simple numeric values
            try:
                clicks = float(data.get(clicks_col, data) if isinstance(data, dict) else data)
                impressions = float(kwargs.get(impressions_col, kwargs.get('impressions', 1)))
                ctr = (clicks / impressions * 100) if impressions > 0 else 0
                return ToolResult(
                    success=True,
                    output={"ctr": ctr, "clicks": clicks, "impressions": impressions},
                    metadata={"operation": "calculate_ctr"}
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Error calculating CTR: {str(e)}"
                )
    
    def _calculate_correlation(self, data: pd.DataFrame, columns: list = None) -> ToolResult:
        """Calculate correlation matrix."""
        if not isinstance(data, pd.DataFrame):
            return ToolResult(
                success=False,
                output=None,
                error="Data must be a DataFrame for correlation calculation"
            )
        
        if columns:
            data = data[columns]
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return ToolResult(
                success=False,
                output=None,
                error="Need at least 2 numeric columns for correlation"
            )
        
        corr_matrix = numeric_data.corr()
        
        return ToolResult(
            success=True,
            output=corr_matrix,
            metadata={"operation": "correlation", "columns": list(corr_matrix.columns)}
        )
    
    def _aggregate(self, data: pd.DataFrame, group_by: str = None, agg_func: str = "mean", columns: list = None) -> ToolResult:
        """Perform aggregation operations."""
        if not isinstance(data, pd.DataFrame):
            return ToolResult(
                success=False,
                output=None,
                error="Data must be a DataFrame for aggregation"
            )
        
        try:
            if group_by:
                if group_by not in data.columns:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Column '{group_by}' not found for grouping"
                    )
                
                grouped = data.groupby(group_by)
                
                if columns:
                    grouped = grouped[columns]
                
                # Map string to pandas function
                agg_map = {
                    "mean": "mean",
                    "sum": "sum",
                    "count": "count",
                    "min": "min",
                    "max": "max",
                    "std": "std"
                }
                
                agg_func_pd = agg_map.get(agg_func.lower(), "mean")
                result = getattr(grouped, agg_func_pd)()
            else:
                # Aggregate entire DataFrame
                if columns:
                    data = data[columns]
                numeric_data = data.select_dtypes(include=[np.number])
                result = getattr(numeric_data, agg_func.lower())()
            
            return ToolResult(
                success=True,
                output=result,
                metadata={"operation": "aggregate", "group_by": group_by, "agg_func": agg_func}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Aggregation error: {str(e)}"
            )
    
    def _percentage_change(self, data: Any, old_value: float = None, new_value: float = None) -> ToolResult:
        """Calculate percentage change."""
        if old_value is not None and new_value is not None:
            change = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
            return ToolResult(
                success=True,
                output={"percentage_change": change, "old_value": old_value, "new_value": new_value},
                metadata={"operation": "percentage_change"}
            )
        elif isinstance(data, dict) and "old_value" in data and "new_value" in data:
            old_val = data["old_value"]
            new_val = data["new_value"]
            change = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
            return ToolResult(
                success=True,
                output={"percentage_change": change, "old_value": old_val, "new_value": new_val},
                metadata={"operation": "percentage_change"}
            )
        else:
            return ToolResult(
                success=False,
                output=None,
                error="Need old_value and new_value for percentage change calculation"
            )
    
    def _statistical_summary(self, data: pd.DataFrame, columns: list = None) -> ToolResult:
        """Generate statistical summary."""
        if not isinstance(data, pd.DataFrame):
            return ToolResult(
                success=False,
                output=None,
                error="Data must be a DataFrame for statistical summary"
            )
        
        if columns:
            data = data[columns]
        
        numeric_data = data.select_dtypes(include=[np.number])
        summary = numeric_data.describe()
        
        return ToolResult(
            success=True,
            output=summary,
            metadata={"operation": "statistical_summary", "columns": list(summary.columns)}
        )
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "operation": {
                "type": "string",
                "description": "Type of operation: 'calculate_ctr', 'correlation', 'aggregate', 'percentage_change', 'statistical_summary'",
                "required": True,
                "enum": ["calculate_ctr", "correlation", "aggregate", "percentage_change", "statistical_summary"]
            },
            "data": {
                "type": "any",
                "description": "Input data (DataFrame, dict, or other depending on operation)",
                "required": False
            },
            "clicks_col": {
                "type": "string",
                "description": "Column name for clicks (for calculate_ctr)",
                "required": False
            },
            "impressions_col": {
                "type": "string",
                "description": "Column name for impressions (for calculate_ctr)",
                "required": False
            },
            "columns": {
                "type": "array",
                "description": "List of column names to operate on",
                "required": False
            },
            "group_by": {
                "type": "string",
                "description": "Column name to group by (for aggregate)",
                "required": False
            },
            "agg_func": {
                "type": "string",
                "description": "Aggregation function: 'mean', 'sum', 'count', 'min', 'max', 'std'",
                "required": False
            },
            "old_value": {
                "type": "number",
                "description": "Old value for percentage_change",
                "required": False
            },
            "new_value": {
                "type": "number",
                "description": "New value for percentage_change",
                "required": False
            }
        }

