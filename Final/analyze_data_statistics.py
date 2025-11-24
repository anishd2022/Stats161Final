"""
Analyze database to extract statistical properties and correlations
for improving synthetic data generation guide
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pymysql
import pandas as pd
import numpy as np
from scipy import stats
import json
from collections import Counter

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load environment variables
env_path = script_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PW = os.getenv('DB_PW')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = int(os.getenv('DB_PORT', 3306))

def get_db_connection(use_dict_cursor=False):
    """Establish a connection to the MySQL database"""
    if use_dict_cursor:
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PW,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
    else:
        # Regular cursor for pandas (pandas has issues with DictCursor)
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PW,
            database=DB_NAME,
            port=DB_PORT
        )

def analyze_numeric_column(df, col):
    """Analyze a numeric column"""
    data = df[col].dropna()
    if len(data) == 0:
        return None
    
    return {
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'q25': float(data.quantile(0.25)),
        'q75': float(data.quantile(0.75)),
        'skewness': float(data.skew()) if len(data) > 2 else 0.0,
        'null_percentage': float(df[col].isna().sum() / len(df) * 100)
    }

def analyze_categorical_column(df, col):
    """Analyze a categorical column"""
    data = df[col].dropna()
    if len(data) == 0:
        return None
    
    value_counts = data.value_counts()
    total = len(data)
    
    return {
        'unique_count': int(data.nunique()),
        'top_10_values': value_counts.head(10).to_dict(),
        'top_10_percentages': {k: float(v/total*100) for k, v in value_counts.head(10).items()},
        'null_percentage': float(df[col].isna().sum() / len(df) * 100),
        'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
        'most_common_pct': float(value_counts.iloc[0] / total * 100) if len(value_counts) > 0 else 0.0
    }

def analyze_json_column(df, col, sample_size=1000):
    """Analyze a JSON/list column"""
    data = df[col].dropna()
    if len(data) == 0:
        return None
    
    # Sample for efficiency
    sample = data.sample(min(sample_size, len(data)))
    
    lengths = []
    element_types = []
    
    for val in sample:
        try:
            if isinstance(val, str):
                parsed = json.loads(val) if val.startswith('[') or val.startswith('{') else val
            else:
                parsed = val
            
            if isinstance(parsed, list):
                lengths.append(len(parsed))
                if parsed:
                    element_types.append(type(parsed[0]).__name__)
            elif isinstance(parsed, dict):
                lengths.append(len(parsed))
            else:
                lengths.append(0)
        except:
            lengths.append(0)
    
    return {
        'avg_length': float(np.mean(lengths)) if lengths else 0.0,
        'median_length': float(np.median(lengths)) if lengths else 0.0,
        'max_length': int(np.max(lengths)) if lengths else 0,
        'min_length': int(np.min(lengths)) if lengths else 0,
        'null_percentage': float(df[col].isna().sum() / len(df) * 100),
        'empty_percentage': float((df[col].isna() | (df[col] == '') | (df[col] == '[]')).sum() / len(df) * 100)
    }

def compute_correlations(df, numeric_cols, max_cols=20):
    """Compute correlations between numeric columns"""
    numeric_df = df[numeric_cols[:max_cols]].select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return {}
    
    corr_matrix = numeric_df.corr()
    
    # Get strong correlations (|r| > 0.3)
    strong_corrs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.3 and not np.isnan(corr_val):
                    strong_corrs.append({
                        'col1': col1,
                        'col2': col2,
                        'correlation': float(corr_val)
                    })
    
    # Sort by absolute correlation
    strong_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return strong_corrs[:30]  # Top 30 correlations

def analyze_label_distribution(df, label_col='label'):
    """Analyze label distribution"""
    if label_col not in df.columns:
        return None
    
    # Try to convert to numeric if it's a string
    label_data = df[label_col].copy()
    if label_data.dtype == 'object':
        try:
            label_data = pd.to_numeric(label_data, errors='coerce')
        except:
            pass
    
    label_counts = label_data.value_counts()
    total = len(label_data.dropna())
    
    if total == 0:
        return None
    
    return {
        'distribution': {str(k): float(v/total*100) for k, v in label_counts.items()},
        'total': int(total),
        'imbalance_ratio': float(label_counts.max() / label_counts.min()) if len(label_counts) > 1 else 1.0
    }

def analyze_table(table_name, connection, sample_size=10000):
    """Analyze a table comprehensively"""
    print(f"\n{'='*80}")
    print(f"Analyzing table: {table_name}")
    print(f"{'='*80}")
    
    # Sample data for analysis
    query = f"SELECT * FROM {table_name} ORDER BY RAND() LIMIT {sample_size}"
    df = pd.read_sql(query, connection)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Try to convert columns to proper types
    # Known numeric columns
    numeric_column_names = {
        'ads': ['log_id', 'label', 'user_id', 'app_score', 'u_refreshTimes', 'u_feedLifeCycle'],
        'feeds': ['u_userId', 'u_refreshTimes', 'u_feedLifeCycle', 'i_dislikeTimes', 'i_upTimes', 'label', 'cillabel', 'pro', 'id']
    }
    
    # Known JSON/list columns
    json_column_names = {
        'ads': ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003', 
                'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003',
                'u_newsCatInterestsST'],
        'feeds': ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST', 
                  'u_click_ca2_news', 'i_entities']
    }
    
    # Convert numeric columns
    for col in numeric_column_names.get(table_name, []):
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Convert JSON columns - try to parse
    for col in json_column_names.get(table_name, []):
        if col in df.columns:
            # Check if it's already a list/dict or if it's a JSON string
            sample = df[col].dropna()
            if len(sample) > 0:
                first_val = sample.iloc[0]
                if isinstance(first_val, str):
                    # Try to detect if it's JSON
                    if first_val.startswith('[') or first_val.startswith('{'):
                        # Mark as JSON but keep as string for now
                        pass
    
    results = {
        'table_name': table_name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': {}
    }
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Also check columns that should be numeric
    for col in numeric_column_names.get(table_name, []):
        if col in df.columns and col not in numeric_cols:
            # Try to convert
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if not converted.isna().all():
                    df[col] = converted
                    numeric_cols.append(col)
            except:
                pass
    
    categorical_cols = []
    json_cols = []
    
    for col in df.columns:
        if col in numeric_cols:
            continue
        if col in json_column_names.get(table_name, []):
            json_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"\nColumn types:")
    print(f"  Numeric: {len(numeric_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")
    print(f"  JSON/List: {len(json_cols)}")
    
    # Analyze numeric columns
    print("\nAnalyzing numeric columns...")
    for col in numeric_cols:
        print(f"  {col}...", end=' ', flush=True)
        stats = analyze_numeric_column(df, col)
        if stats:
            results['columns'][col] = {
                'type': 'numeric',
                'statistics': stats
            }
        print("✓")
    
    # Analyze categorical columns
    print("\nAnalyzing categorical columns...")
    for col in categorical_cols:
        print(f"  {col}...", end=' ', flush=True)
        stats = analyze_categorical_column(df, col)
        if stats:
            results['columns'][col] = {
                'type': 'categorical',
                'statistics': stats
            }
        print("✓")
    
    # Analyze JSON columns
    print("\nAnalyzing JSON/list columns...")
    for col in json_cols:
        print(f"  {col}...", end=' ', flush=True)
        stats = analyze_json_column(df, col)
        if stats:
            results['columns'][col] = {
                'type': 'json',
                'statistics': stats
            }
        print("✓")
    
    # Compute correlations
    print("\nComputing correlations...")
    if len(numeric_cols) > 1:
        correlations = compute_correlations(df, numeric_cols)
        results['correlations'] = correlations
        print(f"  Found {len(correlations)} strong correlations (|r| > 0.3)")
    
    # Analyze label distribution
    label_cols = ['label', 'ads_label', 'feed_label', 'cillabel']
    for label_col in label_cols:
        if label_col in df.columns:
            print(f"\nAnalyzing {label_col} distribution...")
            label_stats = analyze_label_distribution(df, label_col)
            if label_stats:
                results['label_distribution'] = label_stats
                print(f"  Distribution: {label_stats['distribution']}")
                print(f"  Imbalance ratio: {label_stats['imbalance_ratio']:.2f}")
            break
    
    return results

def format_statistics_for_guide(analysis_results):
    """Format analysis results into text for the guide"""
    lines = []
    
    table_name = analysis_results['table_name']
    lines.append(f"\n{table_name.upper()} TABLE - STATISTICAL PROPERTIES")
    lines.append("=" * 80)
    
    # Label distribution
    if 'label_distribution' in analysis_results:
        label_dist = analysis_results['label_distribution']
        lines.append(f"\nLabel Distribution:")
        lines.append(f"  - Total samples: {label_dist['total']:,}")
        for label, pct in label_dist['distribution'].items():
            lines.append(f"  - Label {label}: {pct:.2f}%")
        lines.append(f"  - Imbalance ratio: {label_dist['imbalance_ratio']:.2f}x")
        if label_dist['imbalance_ratio'] > 10:
            lines.append(f"  - WARNING: Highly imbalanced dataset - maintain this ratio in synthetic data")
    
    # Correlations
    if 'correlations' in analysis_results and analysis_results['correlations']:
        lines.append(f"\nStrong Correlations (|r| > 0.3):")
        for corr in analysis_results['correlations'][:15]:  # Top 15
            lines.append(f"  - {corr['col1']} ↔ {corr['col2']}: r = {corr['correlation']:.3f}")
            if abs(corr['correlation']) > 0.7:
                lines.append(f"    → VERY STRONG: These variables should be generated together")
            elif abs(corr['correlation']) > 0.5:
                lines.append(f"    → STRONG: Maintain this relationship in synthetic data")
    
    # Column-specific statistics
    lines.append(f"\nColumn-Specific Statistical Properties:")
    
    for col_name, col_info in sorted(analysis_results['columns'].items()):
        col_type = col_info['type']
        stats = col_info['statistics']
        
        lines.append(f"\n{col_name} ({col_type}):")
        
        if col_type == 'numeric':
            lines.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            lines.append(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}")
            lines.append(f"  Quartiles: Q1={stats['q25']:.2f}, Q3={stats['q75']:.2f}")
            if abs(stats['skewness']) > 1:
                lines.append(f"  Skewness: {stats['skewness']:.2f} ({'right' if stats['skewness'] > 0 else 'left'}-skewed)")
            if stats['null_percentage'] > 0:
                lines.append(f"  Null values: {stats['null_percentage']:.1f}%")
        
        elif col_type == 'categorical':
            lines.append(f"  Unique values: {stats['unique_count']}")
            lines.append(f"  Most common: {stats['most_common']} ({stats['most_common_pct']:.1f}%)")
            if stats['unique_count'] <= 10:
                lines.append(f"  Distribution:")
                for val, pct in list(stats['top_10_percentages'].items())[:5]:
                    lines.append(f"    - {val}: {pct:.1f}%")
            if stats['null_percentage'] > 0:
                lines.append(f"  Null values: {stats['null_percentage']:.1f}%")
        
        elif col_type == 'json':
            lines.append(f"  Average length: {stats['avg_length']:.1f} items")
            lines.append(f"  Length range: [{stats['min_length']}, {stats['max_length']}]")
            lines.append(f"  Median length: {stats['median_length']:.1f} items")
            if stats['empty_percentage'] > 0:
                lines.append(f"  Empty/null: {stats['empty_percentage']:.1f}%")
    
    return "\n".join(lines)

def main():
    """Main analysis function"""
    print("="*80)
    print("Database Statistical Analysis")
    print("="*80)
    
    # Connect to database
    try:
        connection = get_db_connection()
        print("✓ Connected to database")
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return
    
    # Analyze both tables
    all_results = {}
    
    for table_name in ['ads', 'feeds']:
        try:
            results = analyze_table(table_name, connection, sample_size=10000)
            all_results[table_name] = results
        except Exception as e:
            print(f"✗ Error analyzing {table_name}: {e}")
    
    connection.close()
    
    # Format results for guide
    print("\n" + "="*80)
    print("Formatting results for guide...")
    print("="*80)
    
    guide_sections = []
    for table_name in ['ads', 'feeds']:
        if table_name in all_results:
            section = format_statistics_for_guide(all_results[table_name])
            guide_sections.append(section)
    
    # Save to file
    output_file = script_dir / "statistical_analysis_results.txt"
    with open(output_file, 'w') as f:
        f.write("\n\n".join(guide_sections))
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\nNow updating synthetic_data_generation_guide.txt...")
    
    # Read existing guide
    guide_file = script_dir / "rag_docs" / "synthetic_data_generation_guide.txt"
    with open(guide_file, 'r') as f:
        guide_content = f.read()
    
    # Find and replace the old statistics section if it exists
    stats_marker_start = "================================================================================\nSTATISTICAL PROPERTIES (Based on Real Data Analysis)\n================================================================================\n"
    stats_marker_end = "\n\n================================================================================\n"
    
    # Check if old section exists
    if stats_marker_start in guide_content:
        # Find the end of the statistics section (before "ADS TABLE GENERATION GUIDELINES")
        end_marker = "ADS TABLE GENERATION GUIDELINES"
        if end_marker in guide_content:
            # Split at start and end
            before_stats = guide_content.split(stats_marker_start)[0]
            after_stats = guide_content.split(stats_marker_start)[1]
            # Find where the stats section ends (before ADS TABLE GENERATION GUIDELINES)
            if end_marker in after_stats:
                after_stats = after_stats.split(end_marker)[1]
            else:
                # If no end marker, find the next major section
                after_stats = after_stats.split("\n\nADS TABLE GENERATION GUIDELINES")[-1] if "ADS TABLE GENERATION GUIDELINES" in after_stats else after_stats.split("\nADS TABLE GENERATION GUIDELINES")[-1] if "ADS TABLE GENERATION GUIDELINES" in after_stats else after_stats
            
            # Create new statistics section
            stats_section = stats_marker_start + "\n".join(guide_sections) + stats_marker_end
            
            # Reconstruct content
            new_content = before_stats + stats_section + end_marker + after_stats
        else:
            # No end marker, just replace everything after the start marker up to a reasonable point
            parts = guide_content.split(stats_marker_start, 1)
            if len(parts) == 2:
                # Find next major section (look for all caps headers)
                remaining = parts[1]
                # Look for next section starting with all caps
                import re
                match = re.search(r'\n([A-Z][A-Z\s]+\n)', remaining)
                if match:
                    after_stats = remaining[match.start():]
                    stats_section = stats_marker_start + "\n".join(guide_sections) + stats_marker_end
                    new_content = parts[0] + stats_section + after_stats
                else:
                    # Append at end
                    stats_section = stats_marker_start + "\n".join(guide_sections) + stats_marker_end
                    new_content = parts[0] + stats_section
            else:
                new_content = guide_content
    else:
        # No old section, insert before "ADS TABLE GENERATION GUIDELINES"
        insertion_marker = "ADS TABLE GENERATION GUIDELINES"
        if insertion_marker in guide_content:
            parts = guide_content.split(insertion_marker, 1)
            stats_section = "\n\n" + stats_marker_start + "\n".join(guide_sections) + stats_marker_end
            new_content = parts[0] + stats_section + insertion_marker + parts[1]
        else:
            print(f"⚠ Could not find insertion marker in guide file")
            print("  Statistics saved separately to statistical_analysis_results.txt")
            return
    
    # Write updated guide
    with open(guide_file, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated {guide_file}")
    print(f"  Added/updated statistical properties section with {len(guide_sections)} table analyses")

if __name__ == "__main__":
    main()

