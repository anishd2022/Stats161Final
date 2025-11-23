"""
Hybrid Self-Instruct Synthetic Data Generator
Combines LLM-based generation with statistical validation for high-quality synthetic data
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
    # Create dummy stats module
    class DummyStats:
        @staticmethod
        def ks_2samp(a, b):
            return (0.0, 1.0)  # Return dummy values
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
    Hybrid self-instruct synthetic data generator that:
    1. Uses LLM to generate instructions and data
    2. Validates with schema constraints
    3. Validates with statistical properties
    4. Uses LLM self-critique for quality control
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
        
    def sample_seed_data(self, table_name: str, n_samples: int = 100) -> pd.DataFrame:
        """
        Sample seed data from the database.
        
        Args:
            table_name: Name of the table to sample from
            n_samples: Number of samples to retrieve
            
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
                # Try to parse JSON
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
    
    def generate_instructions(self, n_instructions: int = 20, rag_context: str = "") -> List[str]:
        """
        Generate instructions for synthetic data generation using self-instruct.
        
        Args:
            n_instructions: Number of instructions to generate
            rag_context: Additional context from RAG system
            
        Returns:
            List of instruction strings
        """
        if self.seed_data is None:
            raise ValueError("Must sample seed data first")
        
        # Format seed data sample (first 10 rows for context)
        seed_sample = self.seed_data.head(10)
        seed_text = seed_sample.to_string(index=False)
        
        # Format column information
        col_info_text = self._format_column_info()
        
        # Create prompt for instruction generation
        prompt = f"""You are a data generation expert. Your task is to create diverse instructions for generating synthetic data rows that match the structure and patterns of the provided dataset.

Database Schema:
{self.schema}

Column Information:
{col_info_text}

Sample Data (first 10 rows):
{seed_text}
{rag_context}

Your task: Generate {n_instructions} diverse, specific instructions for creating synthetic data rows. Each instruction should describe a different scenario or pattern to generate.

Instructions should:
1. Be specific about what kind of row to generate (e.g., "Generate a record where the user is using a mobile device and clicked on an ad")
2. Cover diverse scenarios (different user types, device types, interaction patterns, etc.)
3. Include edge cases (e.g., "Generate a record with very low advertisement quality score")
4. Reference actual column names from the schema
5. Be realistic and consistent with the data domain

Format: Return ONLY a numbered list of instructions, one per line, like:
1. Generate a record where...
2. Generate a record with...
3. ...

Do not include any other text or explanation. Just the numbered list of instructions."""

        try:
            response = self.model.generate_content(prompt)
            instructions_text = response.text.strip()
            
            # Parse instructions
            instructions = []
            for line in instructions_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering (e.g., "1. ", "2. ", etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                if line:
                    instructions.append(line)
            
            # Ensure we have enough instructions
            if len(instructions) < n_instructions:
                # Generate more if needed
                remaining = n_instructions - len(instructions)
                additional = self.generate_instructions(remaining, rag_context)
                instructions.extend(additional)
            
            print(f"✓ Generated {len(instructions)} instructions")
            return instructions[:n_instructions]
            
        except Exception as e:
            raise Exception(f"Error generating instructions: {e}")
    
    def _format_column_info(self) -> str:
        """Format column information for prompts."""
        lines = []
        for col, info in self.column_info.items():
            line = f"- {col} ({info['type']})"
            if info['type'] == 'numeric':
                if info['min'] is not None and info['max'] is not None:
                    line += f": range [{info['min']:.2f}, {info['max']:.2f}]"
                if info['mean'] is not None:
                    line += f", mean={info['mean']:.2f}"
            elif info['type'] == 'categorical' and 'top_values' in info:
                top_vals = list(info['top_values'].keys())[:3]
                line += f": common values {top_vals}"
            lines.append(line)
        return "\n".join(lines)
    
    def generate_synthetic_rows(self, instructions: List[str], rows_per_instruction: int = 5, 
                                rag_context: str = "") -> List[Dict[str, Any]]:
        """
        Generate synthetic data rows from instructions.
        
        Args:
            instructions: List of instruction strings
            rows_per_instruction: Number of rows to generate per instruction
            rag_context: Additional context from RAG system
            
        Returns:
            List of dictionaries representing synthetic rows
        """
        all_rows = []
        
        for i, instruction in enumerate(instructions):
            try:
                rows = self._generate_rows_from_instruction(
                    instruction, rows_per_instruction, rag_context
                )
                all_rows.extend(rows)
                print(f"  Generated {len(rows)} rows from instruction {i+1}/{len(instructions)}")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to generate rows from instruction {i+1}: {e}")
                continue
        
        print(f"✓ Generated {len(all_rows)} total synthetic rows")
        return all_rows
    
    def generate_synthetic_rows_with_early_stop(self, instructions: List[str], 
                                                rows_per_instruction: int = 5,
                                                target_rows: int = 100,
                                                rag_context: str = "") -> List[Dict[str, Any]]:
        """
        Generate synthetic data rows from instructions with early stopping.
        Stops generating once we have enough rows (accounting for validation filtering).
        
        Args:
            instructions: List of instruction strings
            rows_per_instruction: Number of rows to generate per instruction
            target_rows: Target number of rows to generate (with buffer)
            rag_context: Additional context from RAG system
            
        Returns:
            List of dictionaries representing synthetic rows
        """
        all_rows = []
        
        for i, instruction in enumerate(instructions):
            # Early stop if we have enough rows
            if len(all_rows) >= target_rows:
                print(f"  ✓ Early stop: Generated {len(all_rows)} rows (target: {target_rows})")
                break
            
            try:
                # Calculate how many rows we still need
                remaining = target_rows - len(all_rows)
                rows_to_generate = min(rows_per_instruction, remaining)
                
                if rows_to_generate <= 0:
                    break
                
                rows = self._generate_rows_from_instruction(
                    instruction, rows_to_generate, rag_context
                )
                all_rows.extend(rows)
                print(f"  Generated {len(rows)} rows from instruction {i+1}/{len(instructions)} (total: {len(all_rows)}/{target_rows})")
            except Exception as e:
                print(f"  ⚠ Warning: Failed to generate rows from instruction {i+1}: {e}")
                continue
        
        print(f"✓ Generated {len(all_rows)} total synthetic rows")
        return all_rows
    
    def _generate_rows_from_instruction(self, instruction: str, n_rows: int, 
                                       rag_context: str = "") -> List[Dict[str, Any]]:
        """Generate rows from a single instruction."""
        # Format column information
        col_info_text = self._format_column_info()
        
        # Get column names
        columns = list(self.seed_data.columns)
        
        prompt = f"""You are a data generation expert. Generate {n_rows} synthetic data rows based on the following instruction.

Database Schema:
{self.schema}

Column Information:
{col_info_text}

Instruction: {instruction}
{rag_context}

Requirements:
1. Generate exactly {n_rows} rows
2. Each row must have all columns: {', '.join(columns)}
3. Data types must match the schema (integers, strings, JSON arrays, etc.)
4. Values must be realistic and consistent with the instruction
5. For JSON columns, provide valid JSON arrays or objects
6. For timestamps (pt_d, e_et), use format YYYYMMDDHHMM (e.g., 202205221430)
7. Ensure referential integrity where applicable (e.g., user_id consistency)
8. **CRITICAL**: Provide actual data VALUES, not column names. For example, use actual numbers like 12345 for log_id, not the string "log_id"

Output Format: Return ONLY a JSON array of objects, where each object represents one row.
Each object should have keys matching the column names exactly, with actual data values (not column names).

Example format (with REAL values):
[
  {{"log_id": 12345, "label": 1, "user_id": 67890, "age": 25, ...}},
  {{"log_id": 12346, "label": 0, "user_id": 67891, "age": 30, ...}}
]

**DO NOT** return column names as values. Each value must be actual data (numbers, strings, arrays, etc.).

Do not include any explanation or markdown formatting. Just the JSON array."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = lines[1:]  # Remove first line
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line
                response_text = "\n".join(lines)
            
            # Parse JSON - try multiple approaches
            rows = None
            try:
                rows = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    try:
                        rows = json.loads(json_match.group(0))
                    except:
                        pass
            
            if rows is None:
                raise Exception("Failed to parse JSON from LLM response")
            
            if not isinstance(rows, list):
                rows = [rows]
            
            # Clean and validate rows
            cleaned_rows = []
            for row_idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    print(f"  ⚠ Warning: Row {row_idx} is not a dictionary, skipping")
                    continue
                
                cleaned_row = {}
                for col in columns:
                    if col in row:
                        value = row[col]
                        # Check if value is a column name (common LLM mistake)
                        if isinstance(value, str) and value in columns:
                            # LLM returned column name instead of value - use default
                            print(f"  ⚠ Warning: Row {row_idx}, column {col} has column name as value, using default")
                            cleaned_row[col] = self._get_default_value(col)
                        else:
                            cleaned_row[col] = value
                    else:
                        # Fill with appropriate default
                        cleaned_row[col] = self._get_default_value(col)
                
                cleaned_rows.append(cleaned_row)
            
            if not cleaned_rows:
                raise Exception("No valid rows generated after cleaning")
            
            return cleaned_rows
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise Exception(f"Error generating rows: {e}")
    
    def _get_default_value(self, column: str) -> Any:
        """Get default value for a column based on its type."""
        if column not in self.column_info:
            return None
        
        info = self.column_info[column]
        
        if info['type'] == 'numeric':
            if 'mean' in info and info['mean'] is not None:
                return int(info['mean']) if 'int' in info['dtype'].lower() else info['mean']
            return 0
        elif info['type'] == 'json':
            return []
        elif info['type'] == 'categorical':
            if 'top_values' in info and info['top_values']:
                return list(info['top_values'].keys())[0]
            return ""
        else:
            return None
    
    def validate_schema(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate rows against schema constraints.
        
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
            
            # Check data types
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
                        errors.append(f"{col}: invalid numeric value {value}")
                
                # Check ranges for numeric
                if col_info.get('type') == 'numeric' and is_valid:
                    try:
                        num_val = float(value)
                        if col_info.get('min') is not None and num_val < col_info['min']:
                            errors.append(f"{col}: value {num_val} below minimum {col_info['min']}")
                        if col_info.get('max') is not None and num_val > col_info['max']:
                            errors.append(f"{col}: value {num_val} above maximum {col_info['max']}")
                    except:
                        pass
                
                # Check JSON format
                if col_info.get('type') == 'json':
                    if not isinstance(value, (dict, list)):
                        if isinstance(value, str):
                            try:
                                json.loads(value)
                            except:
                                is_valid = False
                                errors.append(f"{col}: invalid JSON format")
                        else:
                            is_valid = False
                            errors.append(f"{col}: not a valid JSON structure")
            
            if is_valid and not errors:
                valid_rows.append(row)
            else:
                invalid_rows.append({'row': row, 'errors': errors})
        
        print(f"✓ Schema validation: {len(valid_rows)} valid, {len(invalid_rows)} invalid")
        return valid_rows, invalid_rows
    
    def validate_statistics(self, synthetic_df: pd.DataFrame, 
                           reference_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Validate statistical properties of synthetic data.
        
        Args:
            synthetic_df: DataFrame of synthetic data
            reference_df: Reference DataFrame (seed data if None)
            
        Returns:
            Dictionary with validation metrics
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
        
        # Check distribution similarity for numeric columns
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_df.columns:
                synth_values = synthetic_df[col].dropna()
                ref_values = reference_df[col].dropna()
                
                if len(synth_values) > 0 and len(ref_values) > 0:
                    # Kolmogorov-Smirnov test
                    if SCIPY_AVAILABLE:
                        try:
                            ks_stat, p_value = stats.ks_2samp(ref_values, synth_values)
                            metrics['distribution_similarity'][col] = {
                                'ks_statistic': float(ks_stat),
                                'p_value': float(p_value),
                                'similar': p_value > 0.05
                            }
                        except Exception as e:
                            # Fallback: simple mean/std comparison
                            metrics['distribution_similarity'][col] = {
                                'mean_diff': float(abs(synth_values.mean() - ref_values.mean())),
                                'std_diff': float(abs(synth_values.std() - ref_values.std())),
                                'similar': False
                            }
                    else:
                        # Fallback: simple mean/std comparison
                        metrics['distribution_similarity'][col] = {
                            'mean_diff': float(abs(synth_values.mean() - ref_values.mean())),
                            'std_diff': float(abs(synth_values.std() - ref_values.std())),
                            'similar': False
                        }
        
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
        
        # Correlation similarity (for numeric columns)
        if len(numeric_cols) > 1:
            try:
                synth_corr = synthetic_df[numeric_cols].corr()
                ref_corr = reference_df[numeric_cols].corr()
                
                # Compare correlation matrices
                common_cols = set(synth_corr.columns) & set(ref_corr.columns)
                if common_cols:
                    synth_vals = synth_corr.loc[list(common_cols), list(common_cols)].values
                    ref_vals = ref_corr.loc[list(common_cols), list(common_cols)].values
                    
                    # Flatten and compare
                    synth_flat = synth_vals[np.triu_indices_from(synth_vals, k=1)]
                    ref_flat = ref_vals[np.triu_indices_from(ref_vals, k=1)]
                    
                    if len(synth_flat) > 0 and len(ref_flat) > 0:
                        corr_corr = np.corrcoef(synth_flat, ref_flat)[0, 1]
                        metrics['correlation_similarity'] = float(corr_corr) if not np.isnan(corr_corr) else None
            except Exception as e:
                pass
        
        return metrics
    
    def llm_self_critique(self, rows: List[Dict[str, Any]], rag_context: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        Use LLM to critique and filter synthetic rows.
        
        Returns:
            Tuple of (approved_rows, rejected_rows)
        """
        approved_rows = []
        rejected_rows = []
        
        # Batch rows for efficiency
        batch_size = 10
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            
            prompt = f"""You are a data quality expert. Review the following synthetic data rows and determine if they are realistic and consistent with the dataset.

Database Schema:
{self.schema}

Column Information:
{self._format_column_info()}
{rag_context}

Synthetic Rows to Review (JSON format):
{json.dumps(batch, indent=2)}

For each row, evaluate:
1. Are the values realistic and consistent?
2. Do relationships between columns make sense?
3. Are there any obvious errors or inconsistencies?
4. Does the row match the expected data patterns?

Return a JSON object with this structure:
{{
  "approved": [list of row indices (0-based) that are good],
  "rejected": [list of row indices that should be removed],
  "reasons": {{
    "index": "reason for rejection"
  }}
}}

Only reject rows with clear issues. Be lenient - minor variations are acceptable."""

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
                
                # Parse JSON - try multiple approaches
                critique = None
                try:
                    critique = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            critique = json.loads(json_match.group(0))
                        except:
                            pass
                
                if critique is None:
                    # If parsing fails, approve all rows
                    approved_rows.extend(batch)
                    continue
                
                approved_indices = set(critique.get('approved', []))
                rejected_indices = set(critique.get('rejected', []))
                reasons = critique.get('reasons', {})
                
                for idx, row in enumerate(batch):
                    if idx in approved_indices:
                        approved_rows.append(row)
                    elif idx in rejected_indices:
                        rejected_rows.append({
                            'row': row,
                            'reason': reasons.get(str(idx), 'Rejected by LLM critique')
                        })
                    else:
                        # If not explicitly rejected, approve by default
                        approved_rows.append(row)
                        
            except Exception as e:
                print(f"  ⚠ Warning: LLM critique failed for batch {i//batch_size + 1}: {e}")
                # If critique fails, approve all rows in batch
                approved_rows.extend(batch)
        
        print(f"✓ LLM critique: {len(approved_rows)} approved, {len(rejected_rows)} rejected")
        return approved_rows, rejected_rows
    
    def generate(self, table_name: str, n_rows: int = 100, seed_size: int = 100,
                 n_instructions: int = None, rows_per_instruction: int = None,
                 use_rag: bool = True, validate: bool = True) -> Dict[str, Any]:
        """
        Main generation pipeline.
        
        Args:
            table_name: Name of table to generate data for
            n_rows: Target number of rows to generate
            seed_size: Number of seed rows to sample
            n_instructions: Number of instructions to generate (auto-calculated if None)
            rows_per_instruction: Rows per instruction (auto-calculated if None)
            use_rag: Whether to use RAG context
            validate: Whether to perform validation
            
        Returns:
            Dictionary with generated data and metrics
        """
        print("="*80)
        print("Starting Synthetic Data Generation")
        print("="*80)
        
        # Calculate optimal parameters based on target rows
        # Account for validation filtering (assume ~10-20% rejection rate)
        target_with_buffer = int(n_rows * 1.2)  # Generate 20% extra to account for filtering
        
        # Auto-calculate instructions and rows_per_instruction if not provided
        if n_instructions is None or rows_per_instruction is None:
            # For small requests (< 20 rows), use fewer instructions with fewer rows each
            if n_rows <= 20:
                rows_per_instruction = max(2, n_rows // 5)  # 2-4 rows per instruction
                n_instructions = max(3, (target_with_buffer + rows_per_instruction - 1) // rows_per_instruction)
            # For medium requests (20-100 rows), use moderate settings
            elif n_rows <= 100:
                rows_per_instruction = 5
                n_instructions = max(5, (target_with_buffer + rows_per_instruction - 1) // rows_per_instruction)
            # For large requests, use more instructions
            else:
                rows_per_instruction = 5
                n_instructions = max(10, min(30, (target_with_buffer + rows_per_instruction - 1) // rows_per_instruction))
        
        print(f"Target: {n_rows} rows | Generating: ~{n_instructions * rows_per_instruction} rows (with buffer for validation)")
        
        # Step 1: Sample seed data
        print("\n[Step 1] Sampling seed data...")
        seed_df = self.sample_seed_data(table_name, seed_size)
        
        # Step 2: Get RAG context
        rag_context = ""
        if use_rag and self.rag_collection:
            print("\n[Step 2] Retrieving RAG context...")
            query = f"Generate synthetic data for {table_name} table"
            retrieved = retrieve_relevant_context(query, self.rag_collection, top_k=5)
            if retrieved:
                rag_context = format_rag_context(retrieved)
                print("✓ RAG context retrieved")
        
        # Step 3: Generate instructions
        print("\n[Step 3] Generating instructions...")
        instructions = self.generate_instructions(n_instructions, rag_context)
        
        # Step 4: Generate synthetic rows (with early stopping)
        print("\n[Step 4] Generating synthetic rows...")
        synthetic_rows = self.generate_synthetic_rows_with_early_stop(
            instructions, rows_per_instruction, target_with_buffer, rag_context
        )
        
        # Step 5: Schema validation
        if validate:
            print("\n[Step 5] Validating schema...")
            valid_rows, invalid_rows = self.validate_schema(synthetic_rows)
            synthetic_rows = valid_rows
        
        # Step 6: LLM self-critique
        if validate:
            print("\n[Step 6] LLM self-critique...")
            approved_rows, rejected_rows = self.llm_self_critique(synthetic_rows, rag_context)
            synthetic_rows = approved_rows
        
        # Step 7: Convert to DataFrame
        print("\n[Step 7] Converting to DataFrame...")
        
        # Clean and type-convert data before DataFrame creation
        cleaned_rows = []
        for row in synthetic_rows:
            cleaned_row = {}
            for col, value in row.items():
                if col not in self.column_info:
                    cleaned_row[col] = value
                    continue
                
                col_info = self.column_info[col]
                
                # Skip if value is a column name (LLM mistake)
                if isinstance(value, str) and value in self.seed_data.columns:
                    cleaned_row[col] = self._get_default_value(col)
                    continue
                
                # Type conversion based on column info
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
                except (ValueError, TypeError) as e:
                    # If conversion fails, use default
                    print(f"  ⚠ Warning: Failed to convert {col}={value}, using default: {e}")
                    cleaned_row[col] = self._get_default_value(col)
            
            cleaned_rows.append(cleaned_row)
        
        synthetic_df = pd.DataFrame(cleaned_rows)
        
        # Ensure we have the right columns in the right order
        if not synthetic_df.empty:
            # Reorder columns to match seed data
            seed_columns = list(self.seed_data.columns)
            existing_cols = [col for col in seed_columns if col in synthetic_df.columns]
            synthetic_df = synthetic_df[existing_cols]
            
            # Final type conversion for numeric columns
            for col in synthetic_df.columns:
                if col in self.column_info:
                    col_info = self.column_info[col]
                    if col_info.get('type') == 'numeric':
                        try:
                            if 'int' in col_info.get('dtype', '').lower():
                                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0).astype(int)
                            else:
                                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0.0)
                        except Exception as e:
                            print(f"  ⚠ Warning: Could not convert {col} to numeric: {e}")
                            # Use default value for the entire column
                            synthetic_df[col] = self._get_default_value(col)
        
        # Step 8: Statistical validation
        stats_metrics = None
        if validate and not synthetic_df.empty:
            print("\n[Step 8] Statistical validation...")
            stats_metrics = self.validate_statistics(synthetic_df)
        
        # Limit to target number of rows
        if len(synthetic_df) > n_rows:
            synthetic_df = synthetic_df.head(n_rows)
        
        print("\n" + "="*80)
        print(f"✓ Generation complete: {len(synthetic_df)} rows generated")
        print("="*80)
        
        return {
            'data': synthetic_df,
            'seed_data': seed_df,
            'instructions': instructions,
            'statistics': stats_metrics,
            'row_count': len(synthetic_df)
        }

