"""
Pure RAG Synthetic Data Generator
Uses RAG guidelines exclusively - no self-instruct needed
Optimized for token efficiency while maintaining high quality
"""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import google.generativeai as genai

# Try importing scipy for statistical tests (optional)
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    class DummyStats:
        @staticmethod
        def ks_2samp(a, b):
            return (0.0, 1.0)
    stats = DummyStats()
    print("⚠ Warning: scipy not available. Statistical validation will be limited.")

from collections import Counter

# Import RAG system if available
try:
    from rag_system import retrieve_relevant_context, format_rag_context
except ImportError:
    retrieve_relevant_context = None
    format_rag_context = None


class SyntheticDataGenerator:
    """
    Pure RAG synthetic data generator that:
    1. Uses RAG exclusively for comprehensive generation guidelines
    2. Generates data directly from RAG patterns and statistical properties
    3. No self-instruct needed - RAG document is comprehensive enough
    4. Optimized for token efficiency
    """
    
    def __init__(self, model, schema: str, rag_collection=None, db_connection=None):
        """
        Initialize the synthetic data generator.
        
        Args:
            model: Gemini model instance
            schema: Database schema string
            rag_collection: RAG collection for context retrieval
            db_connection: Database connection for seed data
        """
        self.model = model
        self.schema = schema
        self.rag_collection = rag_collection
        self.db_connection = db_connection
        self.seed_data = None
        self.table_name = None
        self.column_info = {}
        self.seed_statistics = {}  # Cached statistics to reduce token usage
        
    def sample_seed_data(self, table_name: str, n_samples: int = 50) -> pd.DataFrame:
        """
        Sample seed data from the database (smaller sample for efficiency).
        
        Args:
            table_name: Name of the table to sample from
            n_samples: Number of samples to retrieve (reduced from 100 to 50)
            
        Returns:
            DataFrame with seed data
        """
        if not self.db_connection:
            raise ValueError("Database connection required for seed data sampling")
        
        self.table_name = table_name
        
        # Sample diverse rows (stratified if possible)
        query = f"""
        SELECT * FROM {table_name}
        ORDER BY RAND()
        LIMIT {n_samples}
        """
        
        try:
            seed_df = pd.read_sql(query, self.db_connection)
            self.seed_data = seed_df
            
            # Extract column information
            self._extract_column_info(seed_df)
            
            # Compute and cache statistics (to avoid sending full seed data)
            self._compute_seed_statistics(seed_df)
            
            print(f"✓ Sampled {len(seed_df)} rows from {table_name}")
            return seed_df
        except Exception as e:
            raise Exception(f"Error sampling seed data: {e}")
    
    def _extract_column_info(self, df: pd.DataFrame):
        """Extract statistical information about columns."""
        for col in df.columns:
            col_data = df[col]
            
            info = {
                'dtype': str(col_data.dtype),
                'nullable': col_data.isna().any(),
                'unique_count': col_data.nunique(),
            }
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                info['min'] = float(col_data.min()) if not col_data.isna().all() else None
                info['max'] = float(col_data.max()) if not col_data.isna().all() else None
                info['mean'] = float(col_data.mean()) if not col_data.isna().all() else None
                info['std'] = float(col_data.std()) if not col_data.isna().all() else None
                info['type'] = 'numeric'
            # For JSON columns
            elif col_data.dtype == 'object' and col_data.apply(lambda x: isinstance(x, (dict, list)) or (isinstance(x, str) and (x.startswith('[') or x.startswith('{')))).any():
                info['type'] = 'json'
                try:
                    sample_json = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
                    if sample_json:
                        if isinstance(sample_json, str):
                            parsed = json.loads(sample_json)
                        else:
                            parsed = sample_json
                        if isinstance(parsed, list):
                            info['json_type'] = 'array'
                            if parsed:
                                info['json_element_type'] = type(parsed[0]).__name__
                        elif isinstance(parsed, dict):
                            info['json_type'] = 'object'
                except:
                    pass
            # For categorical/string columns
            else:
                info['type'] = 'categorical'
                value_counts = col_data.value_counts()
                info['top_values'] = value_counts.head(10).to_dict()
                info['value_distribution'] = dict(value_counts)
            
            self.column_info[col] = info
    
    def _compute_seed_statistics(self, df: pd.DataFrame):
        """Compute and cache seed statistics to reduce token usage."""
        self.seed_statistics = {
            'row_count': len(df),
            'numeric_summary': {},
            'categorical_summary': {},
            'label_distribution': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.seed_statistics['numeric_summary'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in numeric_cols:  # Avoid double-counting
                value_counts = df[col].value_counts(normalize=True)
                self.seed_statistics['categorical_summary'][col] = {
                    'top_5': dict(value_counts.head(5)),
                    'unique_count': df[col].nunique()
                }
        
        # Label distribution
        label_cols = ['label', 'ads_label', 'feed_label', 'cillabel']
        for col in label_cols:
            if col in df.columns:
                self.seed_statistics['label_distribution'][col] = dict(df[col].value_counts(normalize=True))
    
    def _format_column_info_compact(self) -> str:
        """Format column information compactly for prompts (token-efficient)."""
        lines = []
        for col, info in self.column_info.items():
            line = f"- {col} ({info['type']})"
            if info['type'] == 'numeric' and col in self.seed_statistics.get('numeric_summary', {}):
                stats = self.seed_statistics['numeric_summary'][col]
                line += f": range [{stats['min']:.0f}, {stats['max']:.0f}], mean={stats['mean']:.1f}"
            elif info['type'] == 'categorical' and col in self.seed_statistics.get('categorical_summary', {}):
                top_vals = list(self.seed_statistics['categorical_summary'][col]['top_5'].keys())[:3]
                line += f": common values {top_vals}"
            lines.append(line)
        return "\n".join(lines)
    
    def _get_rag_generation_guidelines(self) -> str:
        """Retrieve comprehensive generation guidelines from RAG."""
        if not self.rag_collection:
            return ""
        
        # Retrieve synthetic data generation guide
        query = "synthetic data generation guidelines patterns rules"
        retrieved = retrieve_relevant_context(query, self.rag_collection, top_k=5)
        
        if retrieved:
            guidelines = format_rag_context(retrieved)
            return guidelines
        return ""
    
    def generate_synthetic_rows_rag_heavy(self, n_rows: int, rag_context: str = "") -> List[Dict[str, Any]]:
        """
        Generate synthetic rows using pure RAG approach.
        Generates rows directly from RAG guidelines - no self-instruct needed.
        """
        # Get comprehensive RAG guidelines
        rag_guidelines = self._get_rag_generation_guidelines()
        
        # Format compact column info
        col_info = self._format_column_info_compact()
        
        # Get columns
        columns = list(self.seed_data.columns)
        
        # Calculate batch size for efficiency (generate more rows per call)
        # For small requests, generate all at once; for larger, batch
        if n_rows <= 20:
            batch_size = n_rows
            n_batches = 1
        elif n_rows <= 50:
            batch_size = 20
            n_batches = (n_rows + batch_size - 1) // batch_size
        else:
            batch_size = 25
            n_batches = (n_rows + batch_size - 1) // batch_size
        
        all_rows = []
        
        for batch_num in range(n_batches):
            rows_to_generate = min(batch_size, n_rows - len(all_rows))
            if rows_to_generate <= 0:
                break
            
            # Build prompt using pure RAG guidelines
            prompt = f"""Generate {rows_to_generate} synthetic data rows for the {self.table_name} table.

Database Schema:
{self.schema[:800]}  # Truncated for efficiency

RAG Generation Guidelines (FOLLOW THESE CLOSELY - THIS IS YOUR PRIMARY SOURCE):
{rag_guidelines[:3000]}  # Comprehensive guidelines with statistical properties from RAG

Column Information:
{col_info}

CRITICAL REQUIREMENTS (from RAG guidelines):
1. Generate exactly {rows_to_generate} rows
2. All columns must be present: {', '.join(columns)}
3. **STRICTLY follow RAG guidelines** - they contain real statistical properties, correlations, and value ranges
4. Maintain label distribution from RAG (e.g., ~98% label 0, ~2% label 1 for ads table)
5. Preserve correlations mentioned in RAG (e.g., gender ↔ emui_dev, series_group ↔ emui_dev)
6. Use value ranges from RAG statistical properties
7. Data types must match schema exactly
8. For JSON columns, provide valid JSON arrays (e.g., [1, 2, 3] or ["a", "b"])
9. For timestamps (pt_d, e_et), use format YYYYMMDDHHMM (e.g., 202205221430)
10. **NEVER use column names as values** - use actual data values
11. Maintain realistic relationships (e.g., if label=1, include current task_id in ad_click_list)

Output Format: Return ONLY a JSON array of objects:
[
  {{"log_id": 12345, "label": 1, "user_id": 67890, ...}},
  {{"log_id": 12346, "label": 0, "user_id": 67891, ...}}
]

Do not include markdown, explanations, or code blocks. Just the JSON array."""

            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Remove markdown if present
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    response_text = "\n".join(lines)
                
                # Parse JSON
                rows = None
                try:
                    rows = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        try:
                            rows = json.loads(json_match.group(0))
                        except:
                            pass
                
                if rows is None:
                    print(f"  ⚠ Warning: Failed to parse JSON for batch {batch_num + 1}")
                    continue
                
                if not isinstance(rows, list):
                    rows = [rows]
                
                # Clean and validate rows
                for row_idx, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    
                    cleaned_row = {}
                    for col in columns:
                        if col in row:
                            value = row[col]
                            # Check if value is column name (common mistake)
                            if isinstance(value, str) and value in columns:
                                cleaned_row[col] = self._get_default_value(col)
                            else:
                                cleaned_row[col] = value
                        else:
                            cleaned_row[col] = self._get_default_value(col)
                    
                    all_rows.append(cleaned_row)
                
                print(f"  Generated {len(rows)} rows (batch {batch_num + 1}/{n_batches}, total: {len(all_rows)}/{n_rows})")
                
            except Exception as e:
                print(f"  ⚠ Warning: Failed to generate batch {batch_num + 1}: {e}")
                continue
        
        print(f"✓ Generated {len(all_rows)} total synthetic rows")
        return all_rows
    
    def _get_default_value(self, column: str) -> Any:
        """Get default value for a column based on its type."""
        if column not in self.column_info:
            return None
        
        info = self.column_info[column]
        
        if info['type'] == 'numeric':
            if column in self.seed_statistics.get('numeric_summary', {}):
                stats = self.seed_statistics['numeric_summary'][column]
                return int(stats['mean']) if 'int' in info['dtype'].lower() else stats['mean']
            return 0
        elif info['type'] == 'json':
            return []
        elif info['type'] == 'categorical':
            if column in self.seed_statistics.get('categorical_summary', {}):
                top_vals = list(self.seed_statistics['categorical_summary'][column]['top_5'].keys())
                if top_vals:
                    return top_vals[0]
            return ""
        else:
            return None
    
    def validate_schema(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate rows against schema constraints (lightweight, token-efficient).
        
        Returns:
            Tuple of (valid_rows, invalid_rows)
        """
        valid_rows = []
        invalid_rows = []
        
        required_columns = set(self.seed_data.columns)
        
        for row in rows:
            is_valid = True
            errors = []
            
            # Check all required columns exist
            row_columns = set(row.keys())
            missing = required_columns - row_columns
            if missing:
                is_valid = False
                errors.append(f"Missing columns: {missing}")
            
            # Quick type checking (lightweight)
            for col in required_columns:
                if col not in row:
                    continue
                
                value = row[col]
                col_info = self.column_info.get(col, {})
                
                # Type checking
                if col_info.get('type') == 'numeric':
                    try:
                        if 'int' in col_info.get('dtype', '').lower():
                            int(value)
                        else:
                            float(value)
                    except (ValueError, TypeError):
                        is_valid = False
                        errors.append(f"{col}: invalid numeric value")
                
                # Check JSON format
                if col_info.get('type') == 'json':
                    if not isinstance(value, (dict, list)):
                        if isinstance(value, str):
                            try:
                                json.loads(value)
                            except:
                                is_valid = False
                                errors.append(f"{col}: invalid JSON")
                        else:
                            is_valid = False
                            errors.append(f"{col}: not JSON")
            
            if is_valid and not errors:
                valid_rows.append(row)
            else:
                invalid_rows.append({'row': row, 'errors': errors})
        
        print(f"✓ Schema validation: {len(valid_rows)} valid, {len(invalid_rows)} invalid")
        return valid_rows, invalid_rows
    
    def validate_statistics(self, synthetic_df: pd.DataFrame, 
                           reference_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Validate statistical properties (lightweight version).
        """
        if reference_df is None:
            reference_df = self.seed_data
        
        metrics = {
            'row_count': len(synthetic_df),
            'column_count': len(synthetic_df.columns),
            'distribution_similarity': {},
            'correlation_similarity': None,
            'class_balance': {}
        }
        
        # Check distribution similarity for numeric columns (sample)
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 cols
        
        for col in numeric_cols:
            if col in reference_df.columns:
                synth_values = synthetic_df[col].dropna()
                ref_values = reference_df[col].dropna()
                
                if len(synth_values) > 0 and len(ref_values) > 0:
                    if SCIPY_AVAILABLE:
                        try:
                            ks_stat, p_value = stats.ks_2samp(ref_values, synth_values)
                            metrics['distribution_similarity'][col] = {
                                'ks_statistic': float(ks_stat),
                                'p_value': float(p_value),
                                'similar': p_value > 0.05
                            }
                        except:
                            pass
        
        # Check class balance for label columns
        label_cols = ['label', 'ads_label', 'feed_label', 'cillabel']
        for col in label_cols:
            if col in synthetic_df.columns and col in reference_df.columns:
                synth_dist = synthetic_df[col].value_counts(normalize=True).to_dict()
                ref_dist = reference_df[col].value_counts(normalize=True).to_dict()
                metrics['class_balance'][col] = {
                    'synthetic': synth_dist,
                    'reference': ref_dist
                }
        
        return metrics
    
    def generate(self, table_name: str, n_rows: int = 100, seed_size: int = 50,
                 use_rag: bool = True, validate: bool = True, 
                 use_llm_critique: bool = False) -> Dict[str, Any]:
        """
        Main generation pipeline (pure RAG, token-efficient).
        
        Args:
            table_name: Name of table to generate data for
            n_rows: Target number of rows to generate
            seed_size: Number of seed rows to sample (reduced to 50 for efficiency)
            use_rag: Whether to use RAG context (should be True)
            validate: Whether to perform validation
            use_llm_critique: Whether to use LLM self-critique (optional, adds tokens)
            
        Returns:
            Dictionary with generated data and metrics
        """
        print("="*80)
        print("Starting Pure RAG Synthetic Data Generation")
        print("="*80)
        
        # Calculate target with buffer
        target_with_buffer = int(n_rows * 1.15)  # Smaller buffer (15% vs 20%)
        
        print(f"Target: {n_rows} rows | Generating: ~{target_with_buffer} rows (with buffer)")
        
        # Step 1: Sample seed data (smaller sample)
        print("\n[Step 1] Sampling seed data...")
        seed_df = self.sample_seed_data(table_name, seed_size)
        
        # Step 2: Get RAG context (comprehensive)
        rag_context = ""
        if use_rag and self.rag_collection:
            print("\n[Step 2] Retrieving RAG context...")
            # Get multiple types of context
            contexts = []
            
            # Generation guidelines
            query1 = f"Generate synthetic data for {table_name} table"
            retrieved1 = retrieve_relevant_context(query1, self.rag_collection, top_k=5)
            if retrieved1:
                contexts.append(format_rag_context(retrieved1))
            
            # Variable descriptions
            query2 = f"{table_name} variable descriptions column meanings"
            retrieved2 = retrieve_relevant_context(query2, self.rag_collection, top_k=3)
            if retrieved2:
                contexts.append(format_rag_context(retrieved2))
            
            rag_context = "\n\n".join(contexts)
            if rag_context:
                print("✓ RAG context retrieved")
        
        # Step 3: Generate synthetic rows directly from RAG (pure RAG approach)
        print("\n[Step 3] Generating synthetic rows (pure RAG approach)...")
        synthetic_rows = self.generate_synthetic_rows_rag_heavy(
            target_with_buffer, rag_context
        )
        
        # Step 4: Schema validation (lightweight)
        if validate:
            print("\n[Step 4] Validating schema...")
            valid_rows, invalid_rows = self.validate_schema(synthetic_rows)
            synthetic_rows = valid_rows
        
        # Step 5: Optional LLM self-critique (can be skipped for token savings)
        if validate and use_llm_critique:
            print("\n[Step 5] LLM self-critique (optional)...")
            approved_rows, rejected_rows = self._lightweight_critique(synthetic_rows, rag_context)
            synthetic_rows = approved_rows
        else:
            print("\n[Step 5] Skipping LLM critique (token optimization)")
        
        # Step 6: Convert to DataFrame
        print("\n[Step 6] Converting to DataFrame...")
        cleaned_rows = self._clean_and_convert_rows(synthetic_rows)
        synthetic_df = pd.DataFrame(cleaned_rows)
        
        # Ensure correct column order
        if not synthetic_df.empty:
            seed_columns = list(self.seed_data.columns)
            existing_cols = [col for col in seed_columns if col in synthetic_df.columns]
            synthetic_df = synthetic_df[existing_cols]
            
            # Final type conversion
            for col in synthetic_df.columns:
                if col in self.column_info:
                    col_info = self.column_info[col]
                    if col_info.get('type') == 'numeric':
                        try:
                            if 'int' in col_info.get('dtype', '').lower():
                                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0).astype(int)
                            else:
                                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0.0)
                        except:
                            synthetic_df[col] = self._get_default_value(col)
        
        # Step 7: Statistical validation (lightweight)
        stats_metrics = None
        if validate and not synthetic_df.empty:
            print("\n[Step 7] Statistical validation...")
            stats_metrics = self.validate_statistics(synthetic_df)
        
        # Limit to target
        if len(synthetic_df) > n_rows:
            synthetic_df = synthetic_df.head(n_rows)
        
        print("\n" + "="*80)
        print(f"✓ Generation complete: {len(synthetic_df)} rows generated")
        print("="*80)
        
        return {
            'data': synthetic_df,
            'seed_data': seed_df,
            'statistics': stats_metrics,
            'row_count': len(synthetic_df)
        }
    
    def _clean_and_convert_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and type-convert rows before DataFrame creation."""
        cleaned_rows = []
        for row in rows:
            cleaned_row = {}
            for col, value in row.items():
                if col not in self.column_info:
                    cleaned_row[col] = value
                    continue
                
                col_info = self.column_info[col]
                
                # Skip if value is column name
                if isinstance(value, str) and value in self.seed_data.columns:
                    cleaned_row[col] = self._get_default_value(col)
                    continue
                
                # Type conversion
                try:
                    if col_info.get('type') == 'numeric':
                        if 'int' in col_info.get('dtype', '').lower():
                            cleaned_row[col] = int(float(value)) if value != '' else 0
                        else:
                            cleaned_row[col] = float(value) if value != '' else 0.0
                    elif col_info.get('type') == 'json':
                        if isinstance(value, str):
                            try:
                                cleaned_row[col] = json.loads(value)
                            except:
                                cleaned_row[col] = []
                        elif isinstance(value, (dict, list)):
                            cleaned_row[col] = value
                        else:
                            cleaned_row[col] = []
                    else:
                        cleaned_row[col] = str(value) if value is not None else ''
                except (ValueError, TypeError):
                    cleaned_row[col] = self._get_default_value(col)
            
            cleaned_rows.append(cleaned_row)
        
        return cleaned_rows
    
    def _lightweight_critique(self, rows: List[Dict[str, Any]], rag_context: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        Lightweight LLM critique (optional, can be skipped for token savings).
        Only critiques obvious issues.
        """
        approved_rows = []
        rejected_rows = []
        
        # Larger batch size to reduce calls
        batch_size = 20
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            
            # Very concise prompt
            prompt = f"""Quickly review these {len(batch)} synthetic rows. Only reject rows with OBVIOUS errors (wrong types, invalid JSON, impossible values).

Rows:
{json.dumps(batch[:5], indent=1)}  # Only show first 5 for brevity

Return JSON: {{"approved": [0,1,2...], "rejected": [5,7...]}}
Only reject clear errors. Be lenient."""

            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Parse JSON
                critique = None
                try:
                    critique = json.loads(response_text)
                except:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            critique = json.loads(json_match.group(0))
                        except:
                            pass
                
                if critique is None:
                    approved_rows.extend(batch)
                    continue
                
                approved_indices = set(critique.get('approved', []))
                rejected_indices = set(critique.get('rejected', []))
                
                for idx, row in enumerate(batch):
                    if idx in approved_indices or idx not in rejected_indices:
                        approved_rows.append(row)
                    else:
                        rejected_rows.append({'row': row, 'reason': 'Rejected by lightweight critique'})
                        
            except Exception as e:
                approved_rows.extend(batch)  # Approve on error
        
        print(f"✓ Lightweight critique: {len(approved_rows)} approved, {len(rejected_rows)} rejected")
        return approved_rows, rejected_rows

