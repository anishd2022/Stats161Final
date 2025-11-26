#!/usr/bin/env python3
"""
Analyze synthetic SMOTE and ADASYN CSV files to generate detailed statistics
for MySQL schema creation.

This script reads the synthetic data files and outputs comprehensive information
about data types, ranges, null counts, and other characteristics needed to
create appropriate MySQL table schemas.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_column(df, col_name):
    """Analyze a single column and return detailed statistics."""
    col = df[col_name]
    
    info = {
        'column_name': col_name,
        'dtype': str(col.dtype),
        'non_null_count': col.notna().sum(),
        'null_count': col.isna().sum(),
        'null_percentage': (col.isna().sum() / len(col)) * 100,
        'unique_count': col.nunique(),
    }
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(col):
        info['is_numeric'] = True
        info['min'] = float(col.min()) if col.notna().any() else None
        info['max'] = float(col.max()) if col.notna().any() else None
        info['mean'] = float(col.mean()) if col.notna().any() else None
        info['median'] = float(col.median()) if col.notna().any() else None
        info['std'] = float(col.std()) if col.notna().any() else None
        
        # Check if values are integers
        if pd.api.types.is_integer_dtype(col):
            info['is_integer'] = True
            info['min_int'] = int(col.min()) if col.notna().any() else None
            info['max_int'] = int(col.max()) if col.notna().any() else None
        else:
            info['is_integer'] = False
            # Check if all non-null values are effectively integers
            non_null = col.dropna()
            if len(non_null) > 0:
                info['all_values_integers'] = all(
                    float(x).is_integer() for x in non_null if pd.notna(x)
                )
            else:
                info['all_values_integers'] = False
        
        # Check if all values are non-negative
        non_null = col.dropna()
        if len(non_null) > 0:
            info['all_non_negative'] = (non_null >= 0).all()
            info['all_positive'] = (non_null > 0).all()
        else:
            info['all_non_negative'] = None
            info['all_positive'] = None
            
    else:
        info['is_numeric'] = False
        info['is_integer'] = False
        
        # For string columns, get length statistics
        if pd.api.types.is_string_dtype(col) or col.dtype == 'object':
            str_lengths = col.dropna().astype(str).str.len()
            if len(str_lengths) > 0:
                info['max_string_length'] = int(str_lengths.max())
                info['min_string_length'] = int(str_lengths.min())
                info['mean_string_length'] = float(str_lengths.mean())
            else:
                info['max_string_length'] = None
                info['min_string_length'] = None
                info['mean_string_length'] = None
    
    # Sample values (first 5 non-null unique values)
    non_null_unique = col.dropna().unique()[:5]
    info['sample_values'] = [str(x) for x in non_null_unique]
    
    return info

def suggest_mysql_type(col_info):
    """Suggest MySQL data type based on column analysis."""
    if not col_info['is_numeric']:
        max_len = col_info.get('max_string_length', 255)
        if max_len is None:
            return 'TEXT'
        elif max_len <= 255:
            return f'VARCHAR({max_len})'
        elif max_len <= 65535:
            return 'TEXT'
        elif max_len <= 16777215:
            return 'MEDIUMTEXT'
        else:
            return 'LONGTEXT'
    
    # Numeric types
    if col_info['is_integer'] or col_info.get('all_values_integers', False):
        min_val = col_info.get('min_int') or col_info.get('min')
        max_val = col_info.get('max_int') or col_info.get('max')
        
        if min_val is None or max_val is None:
            return 'INT'
        
        # Check if unsigned
        is_unsigned = col_info.get('all_non_negative', False)
        unsigned_str = ' UNSIGNED' if is_unsigned else ''
        
        # Determine size
        if min_val >= 0 and max_val <= 255:
            return f'TINYINT{unsigned_str}'
        elif min_val >= -128 and max_val <= 127:
            return 'TINYINT'
        elif min_val >= 0 and max_val <= 65535:
            return f'SMALLINT{unsigned_str}'
        elif min_val >= -32768 and max_val <= 32767:
            return 'SMALLINT'
        elif min_val >= 0 and max_val <= 4294967295:
            return f'INT{unsigned_str}'
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return 'INT'
        else:
            return f'BIGINT{unsigned_str}'
    else:
        # Floating point
        # Check decimal places needed
        col = None  # We don't have the actual column here, so we'll use a simple heuristic
        return 'DECIMAL(10,2)'  # Default, can be refined

def analyze_file(filepath, dataset_name):
    """Analyze a CSV file and return comprehensive statistics."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name}")
    print(f"File: {filepath}")
    print(f"{'='*80}\n")
    
    # Read the file
    print("Reading CSV file...")
    try:
        df = pd.read_csv(filepath, nrows=None)  # Read all rows
        print(f"✓ Successfully loaded {len(df):,} rows and {len(df.columns)} columns\n")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None
    
    # Basic file information
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
    print(f"File size: {file_size:.2f} MB")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n")
    
    # Column list
    print(f"Column names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")
    print()
    
    # Detailed column analysis
    print(f"{'='*80}")
    print("DETAILED COLUMN ANALYSIS")
    print(f"{'='*80}\n")
    
    all_column_info = []
    
    for col_name in df.columns:
        col_info = analyze_column(df, col_name)
        all_column_info.append(col_info)
        
        print(f"Column: {col_info['column_name']}")
        print(f"  Data Type (pandas): {col_info['dtype']}")
        print(f"  Non-null count: {col_info['non_null_count']:,} ({100-col_info['null_percentage']:.2f}%)")
        print(f"  Null count: {col_info['null_count']:,} ({col_info['null_percentage']:.2f}%)")
        print(f"  Unique values: {col_info['unique_count']:,}")
        
        if col_info['is_numeric']:
            print(f"  Numeric: Yes")
            if col_info['is_integer'] or col_info.get('all_values_integers', False):
                print(f"  Integer: Yes")
                print(f"  Range: {col_info.get('min_int', col_info.get('min'))} to {col_info.get('max_int', col_info.get('max'))}")
            else:
                print(f"  Integer: No (floating point)")
                print(f"  Range: {col_info['min']:.6f} to {col_info['max']:.6f}")
                print(f"  Mean: {col_info['mean']:.6f}")
                print(f"  Median: {col_info['median']:.6f}")
                print(f"  Std Dev: {col_info['std']:.6f}")
            
            if col_info.get('all_non_negative') is not None:
                print(f"  All non-negative: {col_info['all_non_negative']}")
        else:
            print(f"  Numeric: No")
            if col_info.get('max_string_length') is not None:
                print(f"  Max string length: {col_info['max_string_length']}")
                print(f"  Min string length: {col_info['min_string_length']}")
                print(f"  Mean string length: {col_info['mean_string_length']:.2f}")
        
        print(f"  Sample values: {', '.join(col_info['sample_values'][:5])}")
        
        # MySQL type suggestion
        mysql_type = suggest_mysql_type(col_info)
        print(f"  Suggested MySQL type: {mysql_type}")
        print()
    
    # Summary statistics
    print(f"{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    numeric_cols = [c for c in all_column_info if c['is_numeric']]
    integer_cols = [c for c in numeric_cols if c['is_integer'] or c.get('all_values_integers', False)]
    float_cols = [c for c in numeric_cols if not (c['is_integer'] or c.get('all_values_integers', False))]
    string_cols = [c for c in all_column_info if not c['is_numeric']]
    cols_with_nulls = [c for c in all_column_info if c['null_count'] > 0]
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"  Integer columns: {len(integer_cols)}")
    print(f"  Float columns: {len(float_cols)}")
    print(f"String columns: {len(string_cols)}")
    print(f"Columns with nulls: {len(cols_with_nulls)}")
    print()
    
    # Check for potential primary keys
    print("Potential Primary Key Candidates:")
    for col_info in all_column_info:
        if col_info['unique_count'] == len(df) and col_info['null_count'] == 0:
            print(f"  ✓ {col_info['column_name']} (unique, non-null)")
    print()
    
    # Label column analysis (if exists)
    if 'label' in df.columns:
        label_col = df['label']
        print("Label Column Analysis:")
        print(f"  Value counts:")
        value_counts = label_col.value_counts().sort_index()
        for val, count in value_counts.items():
            pct = (count / len(label_col)) * 100
            print(f"    {val}: {count:,} ({pct:.2f}%)")
        print()
    
    # Return the analysis results
    return {
        'dataset_name': dataset_name,
        'filepath': filepath,
        'row_count': len(df),
        'column_count': len(df.columns),
        'file_size_mb': file_size,
        'columns': all_column_info,
        'dataframe': df  # Include for reference
    }

def main():
    """Main function to analyze both SMOTE and ADASYN files."""
    # Get the script directory (Final folder)
    script_dir = Path(__file__).parent
    
    # File paths
    smote_file = script_dir / "synthetic_train_SMOTE_raw.csv"
    adasyn_file = script_dir / "synthetic_train_ADASYN_raw.csv"
    
    print("="*80)
    print("SYNTHETIC DATA ANALYSIS FOR MYSQL SCHEMA CREATION")
    print("="*80)
    
    results = {}
    
    # Analyze SMOTE file
    if smote_file.exists():
        results['SMOTE'] = analyze_file(smote_file, "SMOTE Synthetic Training Data")
    else:
        print(f"\n✗ File not found: {smote_file}")
    
    # Analyze ADASYN file
    if adasyn_file.exists():
        results['ADASYN'] = analyze_file(adasyn_file, "ADASYN Synthetic Training Data")
    else:
        print(f"\n✗ File not found: {adasyn_file}")
    
    # Comparison section
    if 'SMOTE' in results and 'ADASYN' in results:
        print(f"\n{'='*80}")
        print("COMPARISON: SMOTE vs ADASYN")
        print(f"{'='*80}\n")
        
        smote_info = results['SMOTE']
        adasyn_info = results['ADASYN']
        
        print(f"Row counts:")
        print(f"  SMOTE:  {smote_info['row_count']:,}")
        print(f"  ADASYN: {adasyn_info['row_count']:,}")
        print(f"  Difference: {adasyn_info['row_count'] - smote_info['row_count']:,}")
        print()
        
        print(f"Column counts:")
        print(f"  SMOTE:  {smote_info['column_count']}")
        print(f"  ADASYN: {adasyn_info['column_count']}")
        print()
        
        # Check if columns match
        smote_cols = set([c['column_name'] for c in smote_info['columns']])
        adasyn_cols = set([c['column_name'] for c in adasyn_info['columns']])
        
        if smote_cols == adasyn_cols:
            print("✓ Column names match between SMOTE and ADASYN")
        else:
            print("✗ Column names differ:")
            only_smote = smote_cols - adasyn_cols
            only_adasyn = adasyn_cols - smote_cols
            if only_smote:
                print(f"  Only in SMOTE: {only_smote}")
            if only_adasyn:
                print(f"  Only in ADASYN: {only_adasyn}")
        print()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print("Please copy the output above and paste it back into the chat")
    print("to generate the MySQL schema for both tables.\n")

if __name__ == "__main__":
    main()

