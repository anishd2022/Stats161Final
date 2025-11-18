"""
Data Generator Tool - Generates synthetic data or recommendations based on patterns.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from .base_tool import BaseTool, ToolResult


class DataGeneratorTool(BaseTool):
    """Tool for generating synthetic data or recommendations."""
    
    def __init__(self, sql_tool):
        """
        Initialize data generator tool.
        
        Args:
            sql_tool: SQLTool instance for querying the database
        """
        super().__init__(
            name="generate_data",
            description="Generate synthetic data or recommendations based on user patterns, historical behavior, and constraints. Can generate sample ads, user profiles, or recommendations."
        )
        self.sql_tool = sql_tool
    
    def execute(self, type: str, user_id: int = None, count: int = 5, constraints: Dict[str, Any] = None) -> ToolResult:
        """
        Generate data based on type and constraints.
        
        Args:
            type: Type of data to generate ('recommend_ads', 'sample_ads', 'user_profile')
            user_id: User ID for personalized generation
            count: Number of items to generate
            constraints: Additional constraints (e.g., app_score > 4.0, specific categories)
            
        Returns:
            ToolResult with generated data
        """
        if not type:
            return ToolResult(
                success=False,
                output=None,
                error="Type is required"
            )
        
        type_lower = type.lower()
        
        if type_lower == "recommend_ads" or type_lower == "sample_ads":
            return self._generate_ad_recommendations(user_id, count, constraints)
        elif type_lower == "user_profile":
            return self._generate_user_profile(user_id, constraints)
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown generation type: {type}. Supported: 'recommend_ads', 'sample_ads', 'user_profile'"
            )
    
    def _generate_ad_recommendations(self, user_id: int, count: int, constraints: Dict[str, Any]) -> ToolResult:
        """Generate ad recommendations for a user."""
        if user_id is None:
            return ToolResult(
                success=False,
                output=None,
                error="user_id is required for ad recommendations"
            )
        
        try:
            # Step 1: Get user's historical clicked ads
            query_clicked = f"""
                SELECT DISTINCT 
                    adv_id, task_id, spread_app_id, app_second_class, 
                    app_score, creat_type_cd, adv_prim_id, hispace_app_tags
                FROM ads 
                WHERE user_id = {user_id} AND label = 1
                LIMIT 50
            """
            
            result_clicked = self.sql_tool.execute(query_clicked)
            if not result_clicked.success:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to retrieve user's clicked ads: {result_clicked.error}"
                )
            
            clicked_ads = result_clicked.output
            
            if len(clicked_ads) == 0:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"User {user_id} has no clicked ads. Cannot generate recommendations."
                )
            
            # Step 2: Extract patterns from clicked ads
            # Get most common values
            patterns = {}
            if 'adv_id' in clicked_ads.columns and len(clicked_ads) > 0:
                mode_adv = clicked_ads['adv_id'].mode()
                if len(mode_adv) > 0:
                    patterns['adv_id'] = mode_adv.iloc[0]
            if 'app_second_class' in clicked_ads.columns and len(clicked_ads) > 0:
                mode_app_class = clicked_ads['app_second_class'].mode()
                if len(mode_app_class) > 0:
                    patterns['app_second_class'] = mode_app_class.iloc[0]
            if 'app_score' in clicked_ads.columns and len(clicked_ads) > 0:
                min_score = clicked_ads['app_score'].min()
                patterns['min_app_score'] = float(min_score)
            
            # Step 3: Build recommendation query based on patterns
            where_conditions = []
            
            # Add pattern-based conditions
            if patterns.get('adv_id'):
                where_conditions.append(f"adv_id = {patterns['adv_id']}")
            if patterns.get('app_second_class'):
                where_conditions.append(f"app_second_class = {patterns['app_second_class']}")
            if patterns.get('min_app_score'):
                where_conditions.append(f"app_score >= {patterns['min_app_score']}")
            
            # Add user constraints
            if constraints:
                if 'app_score_min' in constraints:
                    where_conditions.append(f"app_score >= {constraints['app_score_min']}")
                if 'app_second_class' in constraints:
                    where_conditions.append(f"app_second_class = {constraints['app_second_class']}")
                if 'exclude_task_ids' in constraints:
                    exclude_ids = ','.join(map(str, constraints['exclude_task_ids']))
                    where_conditions.append(f"task_id NOT IN ({exclude_ids})")
            
            # Build final query
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query_recommend = f"""
                SELECT DISTINCT 
                    task_id, adv_id, spread_app_id, app_second_class, app_score, 
                    creat_type_cd, adv_prim_id, hispace_app_tags
                FROM ads 
                WHERE {where_clause}
                AND user_id != {user_id}  -- Don't recommend ads user has already seen
                ORDER BY app_score DESC
                LIMIT {count * 2}  -- Get more than needed for variety
            """
            
            result_recommend = self.sql_tool.execute(query_recommend)
            if not result_recommend.success:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to retrieve recommendations: {result_recommend.error}"
                )
            
            recommendations = result_recommend.output
            
            # Take top N
            if len(recommendations) > count:
                recommendations = recommendations.head(count)
            
            return ToolResult(
                success=True,
                output=recommendations,
                metadata={
                    "operation": "generate_ad_recommendations",
                    "user_id": user_id,
                    "count_requested": count,
                    "count_returned": len(recommendations),
                    "patterns_used": patterns
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error generating ad recommendations: {str(e)}"
            )
    
    def _generate_user_profile(self, user_id: int, constraints: Dict[str, Any]) -> ToolResult:
        """Generate a synthetic user profile based on patterns."""
        # This is a placeholder - can be expanded later
        return ToolResult(
            success=False,
            output=None,
            error="User profile generation not yet implemented"
        )
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": {
                "type": "string",
                "description": "Type of data to generate: 'recommend_ads', 'sample_ads', 'user_profile'",
                "required": True,
                "enum": ["recommend_ads", "sample_ads", "user_profile"]
            },
            "user_id": {
                "type": "integer",
                "description": "User ID for personalized generation",
                "required": False
            },
            "count": {
                "type": "integer",
                "description": "Number of items to generate (default: 5)",
                "required": False,
                "default": 5
            },
            "constraints": {
                "type": "object",
                "description": "Additional constraints (e.g., {'app_score_min': 4.0, 'app_category': 123})",
                "required": False
            }
        }

