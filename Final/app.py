"""
Flask web application for the database query chatbot
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import pymysql
import pandas as pd
import uuid
from datetime import datetime

# Try importing RAG system (optional - app can work without it)
try:
    from rag_system import load_rag_collection, retrieve_relevant_context, format_rag_context
except ImportError as e:
    print(f"⚠ Warning: Failed to import rag_system: {e}")
    print("  App will continue without RAG functionality.")
    # Create dummy functions to avoid errors (will be defined globally)
    def _dummy_load_rag_collection():
        return None
    def _dummy_retrieve_relevant_context(*args, **kwargs):
        return []
    def _dummy_format_rag_context(*args, **kwargs):
        return ""
    load_rag_collection = _dummy_load_rag_collection
    retrieve_relevant_context = _dummy_retrieve_relevant_context
    format_rag_context = _dummy_format_rag_context

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load environment variables from .env file (if it exists)
# On Render, use environment variables set in the dashboard instead
env_path = script_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading without specific path (uses default locations)
    load_dotenv()

# Get Google API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Read database configuration from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PW = os.getenv('DB_PW')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = int(os.getenv('DB_PORT', 3306))

# Global variables for schema and RAG collection
schema = None
rag_collection = None
model = None

# Chat session management: session_id -> ChatSession
chat_sessions = {}
# Chat metadata: session_id -> {"title": str, "created_at": timestamp, "message_count": int}
chat_metadata = {}

def get_model():
    """Lazy initialization of Gemini model"""
    global model
    if model is not None:
        return model
    
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set GOOGLE_API_KEY in your environment (Render dashboard > Environment tab)."
        )
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        MODEL_NAME = 'gemini-2.5-flash'
        model = genai.GenerativeModel(MODEL_NAME)
        return model
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini model: {e}")

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

def create_full_sql_prompt(schema, user_query, rag_context=""):
    """Create the full prompt for the first message in a chat session."""
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
- JSON columns can be queried using JSON functions like JSON_EXTRACT, JSON_CONTAINS, JSON_TABLE, etc.
- The ads and feeds tables can be joined using ads.user_id = feeds.u_userId
- There is a view called ads_feeds_joined that joins these tables
- Use proper MySQL syntax and data types

User Question: {user_query}

Instructions:
1. First, determine if this question requires SQL to answer:
   - If the question asks about specific data values, counts, aggregations, or requires querying the database → SQL is needed
   - If the question asks about variable definitions, data structure, general information, or can be answered from documentation → SQL is NOT needed
   - **IMPORTANT**: Recommendation-style questions (e.g., "generate ads user X will click", "recommend ads for user Y", "what ads should we show user Z") CAN be answered with SQL by:
     * Finding items the user has interacted with (clicked ads, viewed content, etc.)
     * Finding other items with similar characteristics (same task_id, advertiser_id, app_id, category, etc.)
     * Returning those similar items as recommendations
     * This does NOT require machine learning - it's a similarity-based recommendation using historical data

2. If SQL is needed:
   - Generate MySQL SQL query(ies) to answer the question
   - For recommendation queries: Find items similar to what the user has interacted with (same task_id, advertiser_id, app_id, etc.)
   - If multiple queries are needed, provide them separated by semicolons or newlines
   - Only return the SQL query(ies), without any additional explanation or markdown formatting
   - Do not include code blocks (```sql or ```) around the query
   - Start your response with "SQL_NEEDED:" followed by the query(ies)

3. If SQL is NOT needed:
   - Provide a clear, helpful answer based on the schema and documentation
   - Start your response with "NO_SQL_NEEDED:" followed by your answer

Reasoning Guidelines:
- Questions asking to "generate", "recommend", "suggest", or "find ads user X will click" should use SQL to:
  1. First, find what the user has clicked/interacted with
  2. Then, find other ads/items with matching or similar attributes (task_id, advertiser_id, app_id, etc.)
  3. Return those similar items as recommendations
- This is a data-driven recommendation approach, not requiring ML models
- Be creative with SQL - you can use subqueries, joins, and JSON functions to find similar items

Remember: Only use SQL when absolutely necessary. If the question can be answered from the schema or documentation alone, provide a direct answer instead."""
    
    return prompt

def create_short_sql_prompt(user_query, rag_context=""):
    """Create a short prompt for subsequent messages in a chat session (context already exists)."""
    rag_section = ""
    if rag_context:
        rag_section = f"\nAdditional Context from Documentation:\n{rag_context}\n"
    
    prompt = f"""User Question: {user_query}{rag_section}

Follow the same process as before: determine if SQL is needed, generate SQL queries if required (starting with "SQL_NEEDED:"), or provide a direct answer (starting with "NO_SQL_NEEDED:"). Use the database schema and context from our previous conversation."""
    
    return prompt

def extract_sql_from_response(response_text):
    """
    Extract SQL queries or direct answer from Gemini's response.
    
    Returns:
        Tuple (is_sql_needed, sql_queries_list, direct_answer)
    """
    response_text = response_text.strip()
    
    # Check if SQL is needed
    if response_text.startswith("NO_SQL_NEEDED:"):
        direct_answer = response_text.replace("NO_SQL_NEEDED:", "").strip()
        return (False, [], direct_answer)
    
    if response_text.startswith("SQL_NEEDED:"):
        sql_text = response_text.replace("SQL_NEEDED:", "").strip()
    else:
        # Try to extract SQL even if marker is missing
        sql_text = response_text
    
    # Remove markdown code blocks if present
    if sql_text.startswith("```"):
        lines = sql_text.split("\n")
        # Remove first line (```sql or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql_text = "\n".join(lines)
    
    # Split by semicolon or newline to get multiple queries
    queries = [q.strip() for q in sql_text.split(";") if q.strip()]
    
    # If no semicolon, try splitting by double newline
    if len(queries) == 1 and "\n\n" in queries[0]:
        queries = [q.strip() for q in queries[0].split("\n\n") if q.strip()]
    
    return (True, queries, "")

def get_db_connection():
    """Establish a connection to the MySQL database"""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PW,
        database=DB_NAME,
        port=DB_PORT,
        cursorclass=pymysql.cursors.DictCursor
    )

def execute_sql_query(connection, sql_query):
    """Execute a SQL query and return results as a DataFrame"""
    cursor = connection.cursor()
    try:
        cursor.execute(sql_query)
        
        # Check if it's a SELECT query
        if sql_query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()
        else:
            # For non-SELECT queries, return rowcount
            connection.commit()
            return cursor.rowcount
    except Exception as e:
        connection.rollback()
        raise Exception(f"Error executing SQL query: {e}")
    finally:
        cursor.close()

def format_results_as_text(results, max_rows=100):
    """Convert query results to a text format for LLM processing."""
    if isinstance(results, pd.DataFrame):
        if len(results) == 0:
            return "No rows returned from the query."
        else:
            if len(results) > max_rows:
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

def generate_combined_natural_language_response(user_query, all_sql_queries, all_results, rag_context="", chat_session=None, is_first_message=False):
    """Generate a combined natural language explanation of all query results using Gemini."""
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
    if is_first_message:
        prompt = f"""You are a data analyst assistant. A user asked a question about a database, and we executed SQL query(ies) to get the results. 

Original User Question: {user_query}
{rag_section}{sql_context}

Query Results:
{all_results_text}

Please provide a clear, concise, and easy-to-understand natural language explanation that answers the user's original question. Use the documentation context to provide richer explanations about variables, data structure, or domain knowledge when relevant. Combine all the results into one unified response. Keep the response brief but informative. If there are specific numbers, values, or patterns in the results, highlight them. If no results were returned, explain what that means in the context of the question. Focus on providing a single, cohesive answer that addresses the user's question directly.

For recommendation-style queries (e.g., "generate ads", "recommend ads", "what ads should we show"):
- Present the recommended items as a clear list if specific items are returned
- Explain why these items were recommended (based on similar attributes to what the user has interacted with)
- If the question asks for a specific number (e.g., "5 sample ads"), limit your response to that number
- Format recommendations in a readable way (numbered list, bullet points, etc.)
- Do NOT include HTML tags or div elements in your response - use plain text formatting

Response:"""
    else:
        # Shorter prompt for subsequent messages
        prompt = f"""User Question: {user_query}{rag_section}

SQL Queries Executed:
{sql_context}

Query Results:
{all_results_text}

Provide a clear, concise natural language explanation answering the user's question. Use the same formatting guidelines as before."""
    
    try:
        if chat_session:
            # Use chat session for all messages (maintains context)
            response = chat_session.send_message(prompt)
            return response.text.strip()
        else:
            # Fallback to direct model call if no chat session
            gemini_model = get_model()
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        return f"Error generating natural language response: {e}"

def create_new_chat():
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    chat_metadata[session_id] = {
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "message_count": 0
    }
    return session_id

def get_or_create_chat_session(session_id):
    """Get existing chat session or create a new one if session_id is None."""
    global chat_sessions
    
    if session_id is None or session_id not in chat_sessions:
        # Create new chat session
        if session_id is None:
            session_id = create_new_chat()
        else:
            # Session ID provided but doesn't exist, create metadata
            if session_id not in chat_metadata:
                chat_metadata[session_id] = {
                    "title": "New Chat",
                    "created_at": datetime.now().isoformat(),
                    "message_count": 0
                }
        
        gemini_model = get_model()
        chat_sessions[session_id] = gemini_model.start_chat(history=[])
    
    return session_id, chat_sessions[session_id]

def update_chat_title(session_id, user_query):
    """Update chat title based on first user query if it's still "New Chat"."""
    if session_id in chat_metadata:
        if chat_metadata[session_id]["title"] == "New Chat" and chat_metadata[session_id]["message_count"] == 1:
            # Generate a short title from the first query (max 50 chars)
            title = user_query[:50]
            if len(user_query) > 50:
                title += "..."
            chat_metadata[session_id]["title"] = title

def process_query(user_query, session_id=None):
    """
    Process a single user query and return the response.
    
    Args:
        user_query: User's question
        session_id: Optional chat session ID for maintaining context
    
    Returns:
        Dictionary with response details
    """
    global schema, rag_collection
    
    if not schema:
        return {
            "success": False,
            "error": "Database schema not loaded. Please check server logs.",
            "answer": None
        }
    
    connection = None
    try:
        # Get or create chat session
        session_id, chat_session = get_or_create_chat_session(session_id)
        
        # Check if this is the first message in the chat
        is_first_message = chat_metadata[session_id]["message_count"] == 0
        
        # Retrieve relevant RAG context
        rag_context = ""
        if rag_collection:
            retrieved_chunks = retrieve_relevant_context(user_query, rag_collection, top_k=3)
            if retrieved_chunks:
                rag_context = format_rag_context(retrieved_chunks)
        
        # Create the prompt - full for first message, short for subsequent
        if is_first_message:
            prompt = create_full_sql_prompt(schema, user_query, rag_context)
        else:
            prompt = create_short_sql_prompt(user_query, rag_context)
        
        # Generate response from Gemini using chat session
        if is_first_message:
            # First message: send to chat session (this initializes the history)
            response = chat_session.send_message(prompt)
        else:
            # Subsequent messages: use chat session (maintains history)
            response = chat_session.send_message(prompt)
        
        # Extract SQL queries or direct answer from response
        is_sql_needed, sql_queries, direct_answer = extract_sql_from_response(response.text)
        
        # Handle case where SQL is not needed (direct answer)
        if not is_sql_needed:
            # Update chat metadata
            chat_metadata[session_id]["message_count"] += 1
            update_chat_title(session_id, user_query)
            
            return {
                "success": True,
                "answer": direct_answer,
                "sql_queries": [],
                "needs_sql": False,
                "session_id": session_id
            }
        
        if not sql_queries:
            return {
                "success": False,
                "error": "No SQL queries found in the response.",
                "answer": None,
                "session_id": session_id
            }
        
        # Connect to database
        connection = get_db_connection()
        
        # Execute each query and store results
        all_results = []
        executed_queries = []
        
        for sql_query in sql_queries:
            try:
                results = execute_sql_query(connection, sql_query)
                all_results.append((sql_query, results))
                executed_queries.append({
                    "query": sql_query,
                    "row_count": len(results) if isinstance(results, pd.DataFrame) else results
                })
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error executing query: {e}",
                    "sql_query": sql_query,
                    "answer": None,
                    "session_id": session_id
                }
        
        # Generate combined natural language response for all results
        if all_results:
            natural_language_response = generate_combined_natural_language_response(
                user_query, sql_queries, all_results, rag_context, chat_session, is_first_message
            )
            
            # Update chat metadata
            chat_metadata[session_id]["message_count"] += 1
            update_chat_title(session_id, user_query)
            
            return {
                "success": True,
                "answer": natural_language_response,
                "sql_queries": executed_queries,
                "needs_sql": True,
                "session_id": session_id
            }
        else:
            return {
                "success": False,
                "error": "No results returned from queries.",
                "answer": None,
                "session_id": session_id
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": None,
            "session_id": session_id if 'session_id' in locals() else None
        }
    finally:
        if connection:
            connection.close()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests from the frontend"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        session_id = data.get('session_id', None)
        
        if not user_query:
            return jsonify({
                "success": False,
                "error": "No query provided"
            }), 400
        
        # Process the query
        result = process_query(user_query, session_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get list of all chat sessions"""
    try:
        chats = []
        for session_id, metadata in chat_metadata.items():
            chats.append({
                "session_id": session_id,
                "title": metadata["title"],
                "created_at": metadata["created_at"],
                "message_count": metadata["message_count"]
            })
        # Sort by created_at descending (newest first)
        chats.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify({"success": True, "chats": chats})
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chats/new', methods=['POST'])
def create_chat():
    """Create a new chat session"""
    try:
        session_id = create_new_chat()
        return jsonify({
            "success": True,
            "session_id": session_id,
            "title": chat_metadata[session_id]["title"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def initialize():
    """Initialize schema and RAG collection"""
    global schema, rag_collection
    
    print("Starting application initialization...")
    
    # Load schema
    try:
        schema = load_schema()
        print("✓ Schema loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Error loading schema: {e}")
        print("  App will continue but queries may fail until schema is available.")
        schema = None
    
    # Load RAG collection
    try:
        rag_collection = load_rag_collection()
        if rag_collection:
            chunk_count = rag_collection.count()
            print(f"✓ RAG system loaded ({chunk_count} chunks indexed)")
        else:
            print("⚠ RAG index not found. App will work without RAG context.")
    except Exception as e:
        print(f"⚠ Warning: Could not load RAG system: {e}")
        print("  App will continue without RAG context.")
        rag_collection = None
    
    # Validate database connection (non-blocking)
    try:
        connection = get_db_connection()
        print("✓ Database connection established")
        connection.close()
    except Exception as e:
        print(f"⚠ Warning: Could not connect to database: {e}")
        print("  App will start, but queries will fail until database is available.")
    
    # Check Gemini API key (non-blocking, won't initialize model yet)
    if not GOOGLE_API_KEY:
        print("⚠ Warning: GOOGLE_API_KEY not found in environment variables.")
        print("  App will start, but queries will fail until GOOGLE_API_KEY is set.")
    else:
        print("✓ GOOGLE_API_KEY found (model will be initialized on first use)")
    
    print("✓ Application initialization complete")

# Initialize on import (but don't crash if it fails)
try:
    initialize()
except Exception as e:
    print(f"⚠ Warning during initialization: {e}")
    print("  App will continue, but some features may not work.")

if __name__ == '__main__':
    # For local development only (gunicorn ignores this)
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

