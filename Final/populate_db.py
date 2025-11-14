import pandas as pd
from pathlib import Path
import json
import os
import pymysql
from dotenv import load_dotenv

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load environment variables from .env file in Final/ folder
env_path = script_dir / ".env"
load_dotenv(env_path)

# Read database configuration from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PW = os.getenv('DB_PW')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = int(os.getenv('DB_PORT', 3306))  # Default to 3306 if not provided

# Build paths to mini CSVs in Final/
ads_path = script_dir / "ads_mini.csv"
feeds_path = script_dir / "feeds_mini.csv"

ads = pd.read_csv(ads_path)
feeds = pd.read_csv(feeds_path)

# print(ads.head())
# print(feeds.head())

# print(ads.info())
# print(feeds.info())

# Function to convert "^" separated string columns to lists
def convert_caret_separated_to_list(df):
    """
    Convert columns with "^" separated values to lists.
    Tries to convert to numeric lists first, falls back to string lists if conversion fails.
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Check if column is object type and contains "^" separator
        if df_copy[col].dtype == 'object':
            # Check if any non-null value contains "^"
            if df_copy[col].astype(str).str.contains('\^', na=False, regex=True).any():
                # Function to split and convert to numeric list if possible
                def split_and_convert(val):
                    if pd.isna(val) or val == '':
                        return []
                    val_str = str(val)
                    if '^' in val_str:
                        parts = [x.strip() for x in val_str.split('^') if x.strip() != '']
                        if not parts:
                            return []
                        
                        # Try to convert all parts to numeric
                        numeric_list = []
                        all_numeric = True
                        
                        for x in parts:
                            try:
                                # Try to convert to numeric (int if no decimal, float if decimal)
                                if '.' in x:
                                    numeric_list.append(float(x))
                                else:
                                    numeric_list.append(int(x))
                            except (ValueError, TypeError):
                                # If any value can't be converted, mark as non-numeric
                                all_numeric = False
                                break
                        
                        # Return numeric list if all values converted successfully
                        if all_numeric:
                            return numeric_list
                        else:
                            # Return as string list if conversion fails
                            return parts
                    return [val]
                
                df_copy[col] = df_copy[col].apply(split_and_convert)
    
    return df_copy

# Apply transformation to ads and feeds
ads = convert_caret_separated_to_list(ads)
feeds = convert_caret_separated_to_list(feeds)

# Print head and info again after transformation
print("\n" + "="*50)
print("After transformation:")
print("="*50 + "\n")

print(ads.head())
print(feeds.head())

print(ads.info())
print(feeds.info())

# Print comprehensive schema information for SQL schema design
print("\n" + "="*80)
print("SCHEMA INFORMATION FOR SQL DESIGN")
print("="*80)

def print_schema_info(df, table_name):
    """Print comprehensive schema information for a dataframe"""
    print(f"\n{'='*80}")
    print(f"TABLE: {table_name}")
    print(f"{'='*80}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"\nColumn Details:")
    print("-" * 80)
    
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Data Type: {df[col].dtype}")
        
        # Check if it's a list/array column
        non_null_col = df[col].dropna()
        sample_val = non_null_col.iloc[0] if not non_null_col.empty else None
        
        if sample_val is not None and isinstance(sample_val, list):
            if len(sample_val) > 0:
                list_element_type = type(sample_val[0]).__name__
                print(f"  Column Type: LIST/ARRAY of {list_element_type}")
                print(f"  Sample list length: {len(sample_val)}")
                print(f"  Sample list values: {sample_val[:5]}")  # First 5 elements
            else:
                print(f"  Column Type: LIST/ARRAY (empty lists)")
        else:
            print(f"  Column Type: {df[col].dtype}")
            if df[col].dtype in ['int64', 'float64']:
                if not non_null_col.empty:
                    print(f"  Min: {df[col].min()}")
                    print(f"  Max: {df[col].max()}")
            elif df[col].dtype == 'object' and sample_val is not None and not isinstance(sample_val, list):
                # String column
                if not non_null_col.empty:
                    max_len = non_null_col.astype(str).str.len().max()
                    print(f"  Max string length: {max_len}")
        
        # Null information
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  Nullable: {'Yes' if null_count > 0 else 'No'}")
        print(f"  Null count: {null_count:,} ({null_pct:.2f}%)")
        
        # Unique values - handle list columns differently
        is_list_column = sample_val is not None and isinstance(sample_val, list)
        if is_list_column:
            # For list columns, convert to tuples to count unique values
            try:
                # Convert lists to tuples for hashing
                def list_to_tuple(val):
                    try:
                        if pd.isna(val):
                            return None
                        if not isinstance(val, list):
                            return val
                        if len(val) == 0:
                            return tuple()  # Empty tuple for empty list
                        # Try to convert to tuple - this will fail if list contains unhashable types
                        return tuple(val)
                    except (TypeError, ValueError):
                        # If conversion fails, try string representation as fallback
                        return str(val) if val is not None else None
                
                unique_series = df[col].apply(list_to_tuple)
                unique_count = unique_series.nunique()
                print(f"  Unique values: {unique_count:,}")
                if unique_count <= 20:
                    unique_vals = unique_series.dropna().unique().tolist()
                    # Show first few unique patterns
                    display_vals = unique_vals[:10]
                    print(f"  Unique list patterns (first 10): {display_vals}")
            except Exception as e:
                # Final fallback: use string representation
                try:
                    unique_count = df[col].astype(str).nunique()
                    print(f"  Unique values (as strings): {unique_count:,}")
                except Exception:
                    print(f"  Unique values: (cannot calculate)")
        else:
            # For non-list columns, use standard nunique
            try:
                unique_count = df[col].nunique()
                print(f"  Unique values: {unique_count:,}")
                if unique_count <= 20:
                    unique_vals = df[col].dropna().unique().tolist()
                    # Try to sort if possible (skip if values are lists or other non-sortable types)
                    try:
                        unique_vals_sorted = sorted(unique_vals)
                        print(f"  Unique values list: {unique_vals_sorted}")
                    except (TypeError, ValueError):
                        # Can't sort (e.g., lists), just show first few
                        print(f"  Unique values (first 10): {unique_vals[:10]}")
            except Exception as e:
                print(f"  Unique values: (cannot calculate)")
        
        # Sample values (non-null)
        non_null_samples = df[col].dropna().head(3).tolist()
        if non_null_samples:
            # Truncate long samples for display
            display_samples = []
            for s in non_null_samples:
                if isinstance(s, list):
                    if len(s) > 10:
                        display_samples.append(f"list({len(s)} items): {s[:5]}...")
                    else:
                        display_samples.append(f"list({len(s)} items): {s}")
                elif isinstance(s, str) and len(str(s)) > 100:
                    display_samples.append(f"{str(s)[:100]}...")
                else:
                    display_samples.append(s)
            print(f"  Sample values: {display_samples}")
        else:
            print(f"  Sample values: (all null)")

# Print schema for ads table
print_schema_info(ads, "ads")

# Print schema for feeds table
print_schema_info(feeds, "feeds")

# Print join key information
print(f"\n{'='*80}")
print("JOIN KEY INFORMATION")
print(f"{'='*80}")

# Find user_id columns
ads_user_cols = [col for col in ads.columns if 'user' in col.lower() or 'userid' in col.lower()]
feeds_user_cols = [col for col in feeds.columns if 'user' in col.lower() or 'userid' in col.lower()]

print(f"\nAds user-related columns: {ads_user_cols}")
print(f"Feeds user-related columns: {feeds_user_cols}")

if ads_user_cols and feeds_user_cols:
    ads_user_col = ads_user_cols[0]
    feeds_user_col = feeds_user_cols[0]
    
    print(f"\nUsing join key: ads.{ads_user_col} = feeds.{feeds_user_col}")
    print(f"Ads unique {ads_user_col}: {ads[ads_user_col].nunique():,}")
    print(f"Feeds unique {feeds_user_col}: {feeds[feeds_user_col].nunique():,}")
    
    # Check overlap
    ads_users = set(ads[ads_user_col].dropna().unique())
    feeds_users = set(feeds[feeds_user_col].dropna().unique())
    overlap = len(ads_users & feeds_users)
    print(f"Overlapping users: {overlap:,}")
    print(f"Users only in ads: {len(ads_users - feeds_users):,}")
    print(f"Users only in feeds: {len(feeds_users - ads_users):,}")

# Print primary key candidates
print(f"\n{'='*80}")
print("PRIMARY KEY CANDIDATES")
print(f"{'='*80}")

def is_list_column(df, col):
    """Check if a column contains lists"""
    non_null = df[col].dropna()
    if non_null.empty:
        return False
    sample_val = non_null.iloc[0]
    return isinstance(sample_val, list)

print(f"\nAds table:")
for col in ads.columns:
    # Skip list columns (can't be primary keys)
    if is_list_column(ads, col):
        continue
    try:
        if ads[col].nunique() == len(ads) and not ads[col].isna().any():
            print(f"  - {col}: UNIQUE and NOT NULL (potential primary key)")
    except Exception:
        # Skip if nunique fails (e.g., unhashable types)
        continue

print(f"\nFeeds table:")
for col in feeds.columns:
    # Skip list columns (can't be primary keys)
    if is_list_column(feeds, col):
        continue
    try:
        if feeds[col].nunique() == len(feeds) and not feeds[col].isna().any():
            print(f"  - {col}: UNIQUE and NOT NULL (potential primary key)")
    except Exception:
        # Skip if nunique fails (e.g., unhashable types)
        continue

# Print summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"\nAds table:")
print(f"  Memory usage: {ads.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
ads_list_cols = []
for col in ads.columns:
    non_null = ads[col].dropna()
    if not non_null.empty and isinstance(non_null.iloc[0], list):
        ads_list_cols.append(col)
print(f"  Columns with lists: {ads_list_cols}")

print(f"\nFeeds table:")
print(f"  Memory usage: {feeds.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
feeds_list_cols = []
for col in feeds.columns:
    non_null = feeds[col].dropna()
    if not non_null.empty and isinstance(non_null.iloc[0], list):
        feeds_list_cols.append(col)
print(f"  Columns with lists: {feeds_list_cols}")

print(f"\n{'='*80}")
print("END OF SCHEMA INFORMATION")
print(f"{'='*80}\n")

# ============================================================================
# DATABASE POPULATION CODE
# ============================================================================

def prepare_dataframe_for_db(df, list_columns):
    """
    Prepare dataframe for database insertion by converting list columns to JSON strings.
    
    Args:
        df: pandas DataFrame
        list_columns: list of column names that contain lists
    
    Returns:
        DataFrame with list columns converted to JSON strings
    """
    df_copy = df.copy()
    
    for col in list_columns:
        if col in df_copy.columns:
            # Convert list columns to JSON strings
            def list_to_json(val):
                # Check for None first (before pd.isna which doesn't work with lists)
                if val is None:
                    return None
                # Check if it's a list before using pd.isna
                if isinstance(val, list):
                    if len(val) == 0:
                        return json.dumps([])
                    return json.dumps(val)
                # For non-list types, check for NaN/None
                try:
                    if pd.isna(val):
                        return None
                except (ValueError, TypeError):
                    # pd.isna might fail for some types, treat as valid value
                    pass
                # If it's already a string, try to parse it as JSON first
                if isinstance(val, str):
                    try:
                        # Try to parse and re-serialize to ensure valid JSON
                        parsed = json.loads(val)
                        return json.dumps(parsed)
                    except (json.JSONDecodeError, TypeError):
                        return json.dumps([val])
                return json.dumps([val])
            
            df_copy[col] = df_copy[col].apply(list_to_json)
    
    # Replace NaN/NaT values with None for proper NULL handling in database
    # Note: list columns are already converted to JSON strings (or None if originally NaN)
    # For non-list columns, convert any remaining NaN values to None
    for col in df_copy.columns:
        if col not in list_columns:
            # Convert NaN values to None for non-list columns
            df_copy[col] = df_copy[col].where(pd.notnull(df_copy[col]), None)
        # List columns (now JSON strings) are already handled by list_to_json function
        # which returns None for NaN values and JSON strings for lists
    
    return df_copy

def get_list_columns(df):
    """Identify columns that contain lists"""
    list_cols = []
    for col in df.columns:
        non_null = df[col].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], list):
            list_cols.append(col)
    return list_cols

# Identify list columns in both dataframes
ads_list_cols = get_list_columns(ads)
feeds_list_cols = get_list_columns(feeds)

print("="*80)
print("PREPARING DATA FOR DATABASE INSERTION")
print("="*80)
print(f"\nAds list columns to convert to JSON: {ads_list_cols}")
print(f"Feeds list columns to convert to JSON: {feeds_list_cols}")

# Prepare dataframes for database insertion
ads_db = prepare_dataframe_for_db(ads, ads_list_cols)
feeds_db = prepare_dataframe_for_db(feeds, feeds_list_cols)

print(f"\nAds dataframe prepared: {len(ads_db)} rows, {len(ads_db.columns)} columns")
print(f"Feeds dataframe prepared: {len(feeds_db)} rows, {len(feeds_db.columns)} columns")

# Verify JSON conversion (sample check)
if ads_list_cols:
    sample_col = ads_list_cols[0]
    print(f"\nSample JSON conversion for {sample_col}:")
    print(f"  Original: {ads[sample_col].iloc[0]}")
    print(f"  JSON: {ads_db[sample_col].iloc[0]}")

def insert_ads_data(connection, df, batch_size=1000):
    """
    Insert ads data into database.
    
    Args:
        connection: Database connection object (placeholder)
        df: Prepared DataFrame with JSON strings for list columns
        batch_size: Number of rows to insert per batch
    """
    # Get column names in the correct order matching the schema
    columns = [
        'log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'city',
        'city_rank', 'series_dev', 'series_group', 'emui_dev', 'device_name',
        'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd',
        'adv_prim_id', 'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id',
        'hispace_app_tags', 'app_second_class', 'app_score',
        'ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003',
        'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003',
        'pt_d', 'u_newsCatInterestsST', 'u_refreshTimes', 'u_feedLifeCycle'
    ]
    
    # Create INSERT statement
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"""
        INSERT INTO ads ({', '.join(columns)})
        VALUES ({placeholders})
    """
    
    # Prepare data as list of tuples
    data = []
    for _, row in df[columns].iterrows():
        row_tuple = tuple(row.values)
        data.append(row_tuple)
    
    # Insert in batches
    cursor = connection.cursor()
    total_rows = len(data)
    
    try:
        for i in range(0, total_rows, batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f"  Inserted batch {i//batch_size + 1}: rows {i+1} to {min(i+batch_size, total_rows)}")
        
        print(f"✓ Successfully inserted {total_rows} rows into ads table")
    except Exception as e:
        connection.rollback()
        print(f"✗ Error inserting ads data: {e}")
        raise
    finally:
        cursor.close()

def insert_feeds_data(connection, df, batch_size=1000):
    """
    Insert feeds data into database.
    
    Args:
        connection: Database connection object (placeholder)
        df: Prepared DataFrame with JSON strings for list columns
        batch_size: Number of rows to insert per batch
    """
    # Get column names in the correct order matching the schema
    # Note: 'id' column is auto-increment, so we don't include it
    columns = [
        'u_userId', 'u_phonePrice', 'u_browserLifeCycle', 'u_browserMode',
        'u_feedLifeCycle', 'u_refreshTimes', 'u_newsCatInterests',
        'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news',
        'i_docId', 'i_s_sourceId', 'i_regionEntity', 'i_cat', 'i_entities',
        'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch', 'e_m', 'e_po',
        'e_pl', 'e_rn', 'e_section', 'e_et', 'label', 'cillabel', 'pro'
    ]
    
    # Create INSERT statement
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"""
        INSERT INTO feeds ({', '.join(columns)})
        VALUES ({placeholders})
    """
    
    # Prepare data as list of tuples
    data = []
    for _, row in df[columns].iterrows():
        row_tuple = tuple(row.values)
        data.append(row_tuple)
    
    # Insert in batches
    cursor = connection.cursor()
    total_rows = len(data)
    
    try:
        for i in range(0, total_rows, batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f"  Inserted batch {i//batch_size + 1}: rows {i+1} to {min(i+batch_size, total_rows)}")
        
        print(f"✓ Successfully inserted {total_rows} rows into feeds table")
    except Exception as e:
        connection.rollback()
        print(f"✗ Error inserting feeds data: {e}")
        raise
    finally:
        cursor.close()

def populate_database():
    """
    Main function to populate the database with ads and feeds data.
    Uses environment variables from .env file for database connection.
    """
    print("\n" + "="*80)
    print("POPULATING DATABASE")
    print("="*80)
    
    # Validate that all required environment variables are set
    required_vars = {'DB_HOST': DB_HOST, 'DB_USER': DB_USER, 'DB_PW': DB_PW, 'DB_NAME': DB_NAME}
    missing_vars = [var for var, value in required_vars.items() if value is None]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please ensure your .env file in the Final/ folder contains all required variables."
        )
    
    print(f"\nConnecting to database:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User: {DB_USER}")
    
    connection = None
    try:
        # Create database connection using environment variables
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PW,
            database=DB_NAME,
            charset='utf8mb4'
        )
        print("✓ Database connection established")
        
        print("\nInserting ads data...")
        insert_ads_data(connection, ads_db, batch_size=1000)
        
        print("\nInserting feeds data...")
        insert_feeds_data(connection, feeds_db, batch_size=1000)
        
        print("\n" + "="*80)
        print("DATABASE POPULATION COMPLETE")
        print("="*80)
        
    except pymysql.Error as e:
        print(f"\n✗ Database error: {e}")
        if connection:
            connection.rollback()
        raise
    except Exception as e:
        print(f"\n✗ Error during database population: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection:
            connection.close()
            print("✓ Database connection closed")

# Uncomment the line below to run database population
populate_database()

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print("\nDatabase connection is configured using environment variables from .env file")
print("To populate the database, uncomment the populate_database() call above and run the script")
print("="*80 + "\n")

