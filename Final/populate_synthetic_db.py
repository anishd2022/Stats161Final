import pandas as pd
from pathlib import Path
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

# Build paths to synthetic CSVs in Final/
smote_path = script_dir / "synthetic_train_SMOTE_raw.csv"
adasyn_path = script_dir / "synthetic_train_ADASYN_raw.csv"

print("="*80)
print("STARTING SYNTHETIC DATA LOADING")
print("="*80)
print(f"\nScript directory: {script_dir}")
print(f"SMOTE CSV path: {smote_path}")
print(f"ADASYN CSV path: {adasyn_path}")

# Check if files exist
if not smote_path.exists():
    raise FileNotFoundError(f"SMOTE CSV file not found: {smote_path}")
if not adasyn_path.exists():
    raise FileNotFoundError(f"ADASYN CSV file not found: {adasyn_path}")

# Check file sizes
smote_size = smote_path.stat().st_size / (1024 * 1024)  # Size in MB
adasyn_size = adasyn_path.stat().st_size / (1024 * 1024)  # Size in MB
print(f"\nFile sizes:")
print(f"  SMOTE CSV: {smote_size:.2f} MB")
print(f"  ADASYN CSV: {adasyn_size:.2f} MB")

print(f"\nReading SMOTE CSV...")
smote_df = pd.read_csv(smote_path)
print(f"✓ SMOTE CSV loaded: {len(smote_df)} rows, {len(smote_df.columns)} columns")

print(f"\nReading ADASYN CSV...")
adasyn_df = pd.read_csv(adasyn_path)
print(f"✓ ADASYN CSV loaded: {len(adasyn_df)} rows, {len(adasyn_df.columns)} columns")

# Display basic information
print(f"\n{'='*80}")
print("DATA SUMMARY")
print(f"{'='*80}")
print(f"\nSMOTE dataset:")
print(f"  Rows: {len(smote_df):,}")
print(f"  Columns: {len(smote_df.columns)}")
print(f"  Memory usage: {smote_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Label distribution:")
if 'label' in smote_df.columns:
    label_counts = smote_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = (count / len(smote_df)) * 100
        print(f"    Label {label}: {count:,} ({pct:.2f}%)")

print(f"\nADASYN dataset:")
print(f"  Rows: {len(adasyn_df):,}")
print(f"  Columns: {len(adasyn_df.columns)}")
print(f"  Memory usage: {adasyn_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Label distribution:")
if 'label' in adasyn_df.columns:
    label_counts = adasyn_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = (count / len(adasyn_df)) * 100
        print(f"    Label {label}: {count:,} ({pct:.2f}%)")

# Verify column names match between datasets
smote_cols = set(smote_df.columns)
adasyn_cols = set(adasyn_df.columns)
if smote_cols == adasyn_cols:
    print(f"\n✓ Column names match between SMOTE and ADASYN datasets")
else:
    print(f"\n⚠ Warning: Column names differ between datasets")
    only_smote = smote_cols - adasyn_cols
    only_adasyn = adasyn_cols - smote_cols
    if only_smote:
        print(f"  Only in SMOTE: {only_smote}")
    if only_adasyn:
        print(f"  Only in ADASYN: {only_adasyn}")

# ============================================================================
# DATABASE POPULATION CODE
# ============================================================================

def prepare_dataframe_for_db(df):
    """
    Prepare dataframe for database insertion by handling NaN values.
    Since synthetic data is already processed, we just need to handle nulls.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with NaN values converted to None for proper NULL handling
    """
    df_copy = df.copy()
    
    # Replace NaN/NaT values with None for proper NULL handling in database
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].where(pd.notnull(df_copy[col]), None)
    
    return df_copy

print("\n" + "="*80)
print("PREPARING DATA FOR DATABASE INSERTION")
print("="*80)

# Prepare dataframes for database insertion
smote_db = prepare_dataframe_for_db(smote_df)
adasyn_db = prepare_dataframe_for_db(adasyn_df)

print(f"\nSMOTE dataframe prepared: {len(smote_db)} rows, {len(smote_db.columns)} columns")
print(f"ADASYN dataframe prepared: {len(adasyn_db)} rows, {len(adasyn_db.columns)} columns")

def insert_smote_data(connection, df, batch_size=1000):
    """
    Insert SMOTE synthetic data into database.
    
    Args:
        connection: Database connection object
        df: Prepared DataFrame
        batch_size: Number of rows to insert per batch
    """
    # Get column names in the correct order matching the schema
    # Note: 'id' column is auto-increment, so we don't include it
    columns = [
        'user_id', 'age', 'gender', 'residence', 'city', 'city_rank',
        'series_dev', 'series_group', 'emui_dev', 'device_name', 'device_size',
        'net_type', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
        'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id',
        'hispace_app_tags', 'app_second_class', 'app_score', 'u_refreshTimes',
        'u_feedLifeCycle', 'hour', 'dayofweek', 'feeds_u_phonePrice',
        'feeds_u_browserLifeCycle', 'feeds_u_browserMode', 'feeds_u_feedLifeCycle',
        'feeds_u_refreshTimes', 'feeds_u_newsCatInterests', 'feeds_u_newsCatDislike',
        'feeds_u_newsCatInterestsST', 'feeds_u_click_ca2_news', 'feeds_label', 'label'
    ]
    
    # Verify all columns exist in dataframe
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in SMOTE dataframe: {missing_cols}")
    
    # Create INSERT statement
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"""
        INSERT INTO synthetic_train_smote ({', '.join(columns)})
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
        
        print(f"✓ Successfully inserted {total_rows:,} rows into synthetic_train_smote table")
    except Exception as e:
        connection.rollback()
        print(f"✗ Error inserting SMOTE data: {e}")
        raise
    finally:
        cursor.close()

def insert_adasyn_data(connection, df, batch_size=1000):
    """
    Insert ADASYN synthetic data into database.
    
    Args:
        connection: Database connection object
        df: Prepared DataFrame
        batch_size: Number of rows to insert per batch
    """
    # Get column names in the correct order matching the schema
    # Note: 'id' column is auto-increment, so we don't include it
    columns = [
        'user_id', 'age', 'gender', 'residence', 'city', 'city_rank',
        'series_dev', 'series_group', 'emui_dev', 'device_name', 'device_size',
        'net_type', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
        'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id',
        'hispace_app_tags', 'app_second_class', 'app_score', 'u_refreshTimes',
        'u_feedLifeCycle', 'hour', 'dayofweek', 'feeds_u_phonePrice',
        'feeds_u_browserLifeCycle', 'feeds_u_browserMode', 'feeds_u_feedLifeCycle',
        'feeds_u_refreshTimes', 'feeds_u_newsCatInterests', 'feeds_u_newsCatDislike',
        'feeds_u_newsCatInterestsST', 'feeds_u_click_ca2_news', 'feeds_label', 'label'
    ]
    
    # Verify all columns exist in dataframe
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in ADASYN dataframe: {missing_cols}")
    
    # Create INSERT statement
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"""
        INSERT INTO synthetic_train_adasyn ({', '.join(columns)})
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
        
        print(f"✓ Successfully inserted {total_rows:,} rows into synthetic_train_adasyn table")
    except Exception as e:
        connection.rollback()
        print(f"✗ Error inserting ADASYN data: {e}")
        raise
    finally:
        cursor.close()

def populate_database():
    """
    Main function to populate the database with SMOTE and ADASYN synthetic data.
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
        
        print("\nInserting SMOTE synthetic data...")
        insert_smote_data(connection, smote_db, batch_size=10000)
        
        print("\nInserting ADASYN synthetic data...")
        insert_adasyn_data(connection, adasyn_db, batch_size=10000)
        
        print("\n" + "="*80)
        print("DATABASE POPULATION COMPLETE")
        print("="*80)
        
        # Verify insertion
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM synthetic_train_smote")
        smote_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM synthetic_train_adasyn")
        adasyn_count = cursor.fetchone()[0]
        cursor.close()
        
        print(f"\nVerification:")
        print(f"  Rows in synthetic_train_smote: {smote_count:,}")
        print(f"  Rows in synthetic_train_adasyn: {adasyn_count:,}")
        print(f"  Expected SMOTE rows: {len(smote_df):,}")
        print(f"  Expected ADASYN rows: {len(adasyn_df):,}")
        
        if smote_count == len(smote_df) and adasyn_count == len(adasyn_df):
            print("✓ All rows inserted successfully!")
        else:
            print("⚠ Warning: Row counts don't match expected values")
        
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

# Run database population
populate_database()

print("\n" + "="*80)
print("SYNTHETIC DATA POPULATION COMPLETE")
print("="*80)
print("\nBoth SMOTE and ADASYN synthetic training data have been loaded into the database.")
print("="*80 + "\n")

