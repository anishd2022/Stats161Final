import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import pymysql
import pandas as pd
from rag_system import load_rag_collection, retrieve_relevant_context, format_rag_context

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load environment variables from .env file in Final/ folder
env_path = script_dir / ".env"
load_dotenv(env_path)

# Get Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Read database configuration from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PW = os.getenv('DB_PW')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = int(os.getenv('DB_PORT', 3306))  # Default to 3306 if not provided

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. "
        "Please ensure your .env file contains GOOGLE_API_KEY variable."
    )

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini model
# Using gemini-2.5-flash or pro which is widely available and faster
# If this doesn't work, the error will show available models
MODEL_NAME = 'gemini-2.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

# Initialize RAG system (will be set in main())
rag_collection = None

def load_schema():
    """Load the SQL schema from schema_clean.sql file"""
    schema_path = script_dir / "schema_clean.sql"
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = f.read()
        return schema
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}\n"
            "Please ensure schema_clean.sql exists in the Final/ folder."
        )

def create_sql_prompt(schema, user_query, rag_context=""):
    """
    Create a prompt that includes the schema, RAG context, and instructions for generating SQL queries.
    The prompt allows Gemini to decide if SQL is needed or if it can answer directly.
    
    Args:
        schema: The SQL schema as a string
        user_query: The user's natural language query
        rag_context: Relevant context from documentation (optional)
    
    Returns:
        Complete prompt string
    """
    rag_section = ""
    if rag_context:
        rag_section = f"""
Additional Context from Documentation:
{rag_context}
"""
    
    prompt = f"""You are a helpful database assistant. A user has asked a question about a database. Your task is to determine if the question requires querying the database with SQL, or if it can be answered using the provided context and documentation.

Database Schema:
{schema}
{rag_section}
Important Notes:
- The database uses MySQL syntax
- JSON columns can be queried using JSON functions like JSON_EXTRACT, JSON_CONTAINS, etc.
- The ads and feeds tables can be joined using ads.user_id = feeds.u_userId
- There is a view called ads_feeds_joined that joins these tables
- Use proper MySQL syntax and data types

User Question: {user_query}

Instructions:
1. First, determine if this question requires SQL to answer:
   - If the question asks about specific data values, counts, aggregations, or requires querying the database → SQL is needed
   - If the question asks about variable definitions, data structure, general information, or can be answered from documentation → SQL is NOT needed

2. If SQL is needed:
   - Generate MySQL SQL query(ies) to answer the question
   - If multiple queries are needed, provide them separated by semicolons or newlines
   - Only return the SQL query(ies), without any additional explanation or markdown formatting
   - Do not include code blocks (```sql or ```) around the query
   - Return only the SQL statement(s)

3. If SQL is NOT needed:
   - Return the text: "NO_SQL_NEEDED"
   - Then provide a clear, helpful answer based on the schema and documentation context

Your response:"""
    
    return prompt

def extract_sql_from_response(response_text):
    """
    Extract SQL queries from Gemini's response.
    Returns tuple: (is_sql_needed: bool, sql_queries: list, direct_answer: str)
    """
    text = response_text.strip()
    
    # Check if SQL is not needed
    if "NO_SQL_NEEDED" in text.upper():
        # Extract the direct answer (everything after NO_SQL_NEEDED)
        parts = text.split("NO_SQL_NEEDED", 1)
        direct_answer = parts[1].strip() if len(parts) > 1 else text
        # Clean up the direct answer
        direct_answer = direct_answer.replace("NO_SQL_NEEDED", "").strip()
        return (False, [], direct_answer)
    
    # Remove markdown code blocks if present
    if text.startswith('```sql'):
        text = text[6:]  # Remove ```sql
    elif text.startswith('```'):
        text = text[3:]   # Remove ```
    
    if text.endswith('```'):
        text = text[:-3]  # Remove closing ```
    
    text = text.strip()
    
    # Split by semicolons to handle multiple queries
    # Filter out empty queries
    queries = [q.strip() for q in text.split(';') if q.strip()]
    
    return (True, queries, None)

def get_db_connection():
    """
    Create and return a database connection.
    """
    # Validate that all required environment variables are set
    required_vars = {'DB_HOST': DB_HOST, 'DB_USER': DB_USER, 'DB_PW': DB_PW, 'DB_NAME': DB_NAME}
    missing_vars = [var for var, value in required_vars.items() if value is None]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please ensure your .env file contains all required database variables."
        )
    
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PW,
        database=DB_NAME,
        charset='utf8mb4'
    )
    
    return connection

def execute_sql_query(connection, sql_query):
    """
    Execute a SQL query and return results as a pandas DataFrame.
    
    Args:
        connection: Database connection object
        sql_query: SQL query string to execute
    
    Returns:
        pandas DataFrame with query results, or None for non-SELECT queries
    """
    cursor = connection.cursor()
    
    try:
        # Execute the query
        cursor.execute(sql_query)
        
        # Check if this is a SELECT query (has results)
        if cursor.description:
            # Fetch results and column names
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
            connection.commit()
            rows_affected = cursor.rowcount
            return rows_affected
            
    except Exception as e:
        connection.rollback()
        raise Exception(f"Error executing SQL query: {e}")
    finally:
        cursor.close()

def format_results_as_text(results, max_rows=100):
    """
    Convert query results to a text format for LLM processing.
    
    Args:
        results: pandas DataFrame or integer (for non-SELECT queries)
        max_rows: Maximum number of rows to include (to avoid token limits)
    
    Returns:
        String representation of the results
    """
    if isinstance(results, pd.DataFrame):
        if len(results) == 0:
            return "No rows returned from the query."
        else:
            # Limit the number of rows to avoid token limits
            if len(results) > max_rows:
                # Show first max_rows rows and indicate truncation
                truncated_results = results.head(max_rows)
                results_text = truncated_results.to_string(index=False)
                results_text += f"\n\n[Note: Showing first {max_rows} of {len(results)} total rows]"
                return results_text
            else:
                return results.to_string(index=False)
    elif isinstance(results, int):
        return f"Query executed successfully. {results} row(s) affected."
    else:
        return str(results)

def generate_combined_natural_language_response(user_query, all_sql_queries, all_results, rag_context=""):
    """
    Generate a combined natural language explanation of all query results using Gemini.
    
    Args:
        user_query: Original natural language query from the user
        all_sql_queries: List of all SQL queries that were generated/executed
        all_results: List of tuples (sql_query, results) for all queries
        rag_context: Relevant context from documentation (optional)
    
    Returns:
        Combined natural language explanation of all results
    """
    # Format all SQL queries as context
    if len(all_sql_queries) > 1:
        all_queries_text = "\n\n".join([f"Query {i+1}:\n{q}" for i, q in enumerate(all_sql_queries)])
        sql_context = f"""All SQL Queries Executed:
{all_queries_text}"""
    else:
        sql_context = f"""SQL Query Executed:
{all_sql_queries[0]}"""
    
    # Format all results
    if len(all_results) > 1:
        results_sections = []
        for i, (sql_query, results) in enumerate(all_results, 1):
            results_text = format_results_as_text(results)
            results_sections.append(f"Results from Query {i}:\n{results_text}")
        all_results_text = "\n\n" + "\n\n".join(results_sections)
    else:
        sql_query, results = all_results[0]
        all_results_text = format_results_as_text(results)
    
    # Add RAG context if available
    rag_section = ""
    if rag_context:
        rag_section = f"""
Additional Context from Documentation:
{rag_context}

"""
    
    # Create prompt for natural language formatting
    prompt = f"""You are a data analyst assistant. A user asked a question about a database, and we executed SQL query(ies) to get the results. 

Original User Question: {user_query}
{rag_section}{sql_context}

Query Results:
{all_results_text}

Please provide a clear, concise, and easy-to-understand natural language explanation that answers the user's original question. Use the documentation context to provide richer explanations about variables, data structure, or domain knowledge when relevant. Combine all the results into one unified response. Keep the response brief but informative. If there are specific numbers, values, or patterns in the results, highlight them. If no results were returned, explain what that means in the context of the question. Focus on providing a single, cohesive answer that addresses the user's question directly.

Response:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating natural language response: {e}"

def print_query_results(results, query_num=None, show_raw=False):
    """
    Print query results in a formatted way.
    
    Args:
        results: pandas DataFrame or integer (for non-SELECT queries)
        query_num: Optional query number for multiple queries
        show_raw: If True, also print raw results table
    """
    if query_num is not None:
        print(f"\n{'='*80}")
        print(f"Query {query_num} Results:")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Query Results:")
        print("="*80)
    
    if isinstance(results, pd.DataFrame):
        if len(results) == 0:
            print("(No rows returned)")
        else:
            if show_raw:
                # Print DataFrame with formatting
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(results.to_string(index=False))
                print(f"\n({len(results)} row(s) returned)")
            else:
                print(f"({len(results)} row(s) returned)")
    elif isinstance(results, int):
        print(f"Query executed successfully. {results} row(s) affected.")
    else:
        print(results)
    
    print("="*80)

def process_query(schema, user_query, rag_collection):
    """
    Process a single user query: retrieve context, generate SQL or direct answer, execute if needed.
    
    Args:
        schema: Database schema string
        user_query: User's question
        rag_collection: RAG collection for context retrieval
    
    Returns:
        None (prints results directly)
    """
    connection = None
    try:
        # Retrieve relevant RAG context
        rag_context = ""
        if rag_collection:
            retrieved_chunks = retrieve_relevant_context(user_query, rag_collection, top_k=3)
            if retrieved_chunks:
                rag_context = format_rag_context(retrieved_chunks)
        
        # Create the prompt with schema, RAG context, and user query
        prompt = create_sql_prompt(schema, user_query, rag_context)
        
        # Generate response from Gemini
        print("\nAnalyzing query and generating response...")
        response = model.generate_content(prompt)
        
        # Extract SQL queries or direct answer from response
        is_sql_needed, sql_queries, direct_answer = extract_sql_from_response(response.text)
        
        # Handle case where SQL is not needed (direct answer)
        if not is_sql_needed:
            print("\n" + "="*80)
            print("Answer:")
            print("="*80)
            print(direct_answer)
            print("="*80 + "\n")
            return
        
        if not sql_queries:
            print("No SQL queries found in the response.")
            return
        
        # Print the SQL query(ies)
        print("\n" + "="*80)
        print("Generated SQL Query(ies):")
        print("="*80)
        for i, query in enumerate(sql_queries, 1):
            if len(sql_queries) > 1:
                print(f"\n--- Query {i} ---")
            print(query)
        print("="*80)
        
        # Connect to database
        connection = get_db_connection()
        
        # Execute each query and print results
        print("\nExecuting query(ies)...")
        
        # Store results for natural language formatting
        all_results = []
        
        for i, sql_query in enumerate(sql_queries, 1):
            try:
                results = execute_sql_query(connection, sql_query)
                all_results.append((sql_query, results))
                
                # Print raw results (brief summary)
                print_query_results(results, query_num=i if len(sql_queries) > 1 else None, show_raw=False)
            except Exception as e:
                print(f"\n✗ Error executing query {i}: {e}")
                print(f"Query: {sql_query}")
                # Continue with next query if multiple queries
                if len(sql_queries) > 1:
                    continue
                else:
                    raise
        
        print("\n✓ Query execution complete")
        
        # Generate combined natural language response for all results
        if all_results:
            print("\n" + "="*80)
            print("Generating natural language explanation...")
            print("="*80)
            
            # Generate one combined natural language response for all queries
            # Include RAG context for richer explanations
            natural_language_response = generate_combined_natural_language_response(
                user_query, sql_queries, all_results, rag_context
            )
            
            print("\n" + "="*80)
            print("Answer:")
            print("="*80)
            print(natural_language_response)
            print("="*80 + "\n")
        
    except Exception as e:
        error_msg = str(e)
        
        # If model not found, list available models
        if "404" in error_msg or "not found" in error_msg.lower():
            print("\nAttempting to list available models...")
            try:
                print("Available models that support generateContent:")
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        print(f"  - {m.name}")
                print(f"\nCurrent model: {MODEL_NAME}")
                print("You can update MODEL_NAME in the code to use one of the available models above.")
            except Exception as list_error:
                print(f"Could not list models: {list_error}")
        else:
            print(f"\nError: {error_msg}")
        
        raise
    finally:
        if connection:
            connection.close()

def main():
    """Main function to handle user queries in a loop"""
    global rag_collection
    
    # Load the database schema
    try:
        schema = load_schema()
        print("✓ Schema loaded successfully")
    except Exception as e:
        print(f"Error loading schema: {e}")
        return
    
    # Load RAG collection (must be built first with build_rag_index.py)
    try:
        rag_collection = load_rag_collection()
        if rag_collection:
            chunk_count = rag_collection.count()
            print(f"✓ RAG system loaded ({chunk_count} chunks indexed)")
        else:
            print("⚠ RAG index not found. Run build_rag_index.py to create it.")
            print("Continuing without RAG context...")
    except Exception as e:
        print(f"Warning: Could not load RAG system: {e}")
        print("Run build_rag_index.py to create the index, or continue without RAG context.")
        rag_collection = None
    
    # Validate database connection
    try:
        connection = get_db_connection()
        print("✓ Database connection established")
        connection.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure your database is running and .env file contains correct credentials.")
        return
    
    print("\n" + "="*80)
    print("Query System Ready")
    print("="*80)
    print("Type 'quit' or 'exit' to end the session\n")
    
    # Main query loop
    while True:
        try:
            print("Enter your query: ", end='', flush=True)
            user_query = input().strip()
            
            # Check for quit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_query:
                print("No query provided. Please enter a question or type 'quit' to exit.")
                continue
            
            # Process the query
            process_query(schema, user_query, rag_collection)
            
            # Ask if user wants to continue
            while True:
                print("Do you want to continue? (y / n): ", end='', flush=True)
                continue_response = input().strip().lower()
                
                if continue_response in ['n', 'no']:
                    print("\nGoodbye!")
                    return
                elif continue_response in ['y', 'yes']:
                    print()  # Add blank line before next query
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            # Ask if user wants to continue after error
            while True:
                print("Do you want to continue? (y / n): ", end='', flush=True)
                continue_response = input().strip().lower()
                
                if continue_response in ['n', 'no']:
                    print("\nGoodbye!")
                    return
                elif continue_response in ['y', 'yes']:
                    print()  # Add blank line before next query
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()

