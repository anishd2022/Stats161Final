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
import json
import re
import threading
from datetime import datetime
from typing import Optional, Dict, Any

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

# Try importing synthetic data generator (optional)
try:
    from synthetic_generator import SyntheticDataGenerator
except ImportError as e:
    print(f"⚠ Warning: Failed to import synthetic_generator: {e}")
    print("  Synthetic data generation will not be available.")
    SyntheticDataGenerator = None

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
        # Set temperature to 0.1 for more deterministic responses (better for SQL generation)
        generation_config = genai.types.GenerationConfig(temperature=0.3)
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
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

def detect_synthetic_data_request(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Detect if the user query is asking for synthetic data generation.
    Only triggers on explicit generation requests, NOT on questions about the process.
    
    Returns:
        Dictionary with parsed parameters if synthetic data request detected, None otherwise
    """
    query_lower = user_query.lower().strip()
    
    # Exclude questions about the process (these should be answered, not trigger generation)
    question_words = ['what', 'how', 'explain', 'describe', 'tell me', 'show me', 'can you explain', 
                     'what is', 'what are', 'what kind of', 'how does', 'how do', 'why', 'when']
    
    # If the query starts with a question word, it's likely asking about the process, not requesting generation
    if any(query_lower.startswith(qw) for qw in question_words):
        return None
    
    # Action verbs that indicate actual generation requests
    action_verbs = ['generate', 'create', 'make', 'produce', 'build', 'construct']
    
    # Keywords that indicate synthetic data generation
    synthetic_keywords = [
        'synthetic data', 'synthetic rows', 'synthetic records',
        'synthetic dataset', 'fake data', 'artificial data',
        'new data', 'synthetic sample'
    ]
    
    # Must have BOTH an action verb AND synthetic keywords to be a generation request
    has_action = any(verb in query_lower for verb in action_verbs)
    has_synthetic = any(keyword in query_lower for keyword in synthetic_keywords)
    
    if not (has_action and has_synthetic):
        return None
    
    # Extract table name
    table_name = None
    if 'ads' in query_lower or 'ad' in query_lower:
        if 'feed' not in query_lower or query_lower.find('ads') < query_lower.find('feed'):
            table_name = 'ads'
    if 'feed' in query_lower and table_name is None:
        table_name = 'feeds'
    
    # Default to ads if not specified
    if table_name is None:
        table_name = 'ads'
    
    # Extract number of rows
    n_rows = 100  # default
    row_patterns = [
        r'(\d+)\s*(?:rows?|records?|samples?|data points?)',
        r'generate\s+(\d+)',
        r'create\s+(\d+)',
        r'(\d+)\s*synthetic'
    ]
    
    for pattern in row_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                n_rows = int(match.group(1))
                # Cap at reasonable limit
                n_rows = min(max(n_rows, 1), 10000)
                break
            except:
                pass
    
    return {
        'is_synthetic': True,
        'table_name': table_name,
        'n_rows': n_rows
    }

MATH_KEYWORDS = [
    "statistical test",
    "significant",
    "significance",
    "hypothesis test",
    "p-value",
    "p value",
    "ratio",
    "percentage",
    "percent",
    "average",
    "median",
    "mean",
    "variance",
    "standard deviation",
    "compare",
    "difference",
    "correlation",
    "regression",
    "trend",
    "distribution",
    "math",
    "calculate",
    "calculation",
    "estimate",
    "probability"
]

def detect_math_intensive_request(user_query: str) -> Optional[str]:
    """
    Detect if the user query likely requires multi-step reasoning
    (SQL data retrieval followed by mathematical/statistical analysis).
    
    Returns an instruction string for the LLM if math-heavy, else None.
    """
    text = user_query.lower()
    if not any(keyword in text for keyword in MATH_KEYWORDS):
        return None
    
    return (
        "IMPORTANT: This question requires mathematical or statistical reasoning. "
        "Identify the base metrics you need, generate SQL to retrieve them, and "
        "treat the request as SQL_NEEDED even if additional analysis is required. "
        "Assume you'll be given the SQL results afterward so you can perform the "
        "necessary calculations yourself."
    )

def create_full_sql_prompt(schema, user_query, rag_context="", extra_instructions=""):
    """Create the full prompt for the first message in a chat session."""
    rag_section = ""
    if rag_context:
        rag_section = f"""
Additional Context from Documentation:
{rag_context}
"""
    guidance_section = ""
    if extra_instructions:
        guidance_section = f"""
Additional Guidance:
{extra_instructions}
"""
    
    prompt = f"""You are a helpful database assistant. A user has asked a question about a database. Your task is to determine the type of request and respond appropriately.

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
1. First, determine the type of request:
   a) **SYNTHETIC_DATA_NEEDED**: ONLY if the question explicitly requests to GENERATE/CREATE/MAKE synthetic data with action verbs like "generate", "create", "make". Examples: "generate synthetic data", "create 50 synthetic rows", "make synthetic dataset". 
   
   **CRITICAL**: Questions about synthetic data (like "what process do you use", "how do you generate", "explain synthetic data generation") are NOT generation requests - they are NO_SQL_NEEDED and should be answered with information about the process.
   
   b) **SQL_NEEDED**: If the question asks about specific data values, counts, aggregations, or requires querying the database → SQL is needed
   c) **NO_SQL_NEEDED**: If the question asks about variable definitions, data structure, general information, HOW synthetic data generation works, WHAT process is used, or can be answered from documentation → SQL is NOT needed
   
   **IMPORTANT DISTINCTIONS**:
   - "Generate synthetic data" / "Create synthetic rows" = SYNTHETIC_DATA_NEEDED (creating new artificial data)
   - "What process do you use to generate synthetic data?" / "How do you generate synthetic data?" = NO_SQL_NEEDED (explaining the process)
   - "Recommend ads" / "Suggest ads" / "What ads should I show" = SQL_NEEDED (finding existing similar ads)
   - "Generate ads user X will click" (when asking for recommendations) = SQL_NEEDED (finding similar existing ads)
   - "Randomly sample 100 rows" / "Calculate mean of sample" = SQL_NEEDED (use subquery for sampling first)

2. If SYNTHETIC_DATA_NEEDED:
   - Start your response with "SYNTHETIC_DATA_NEEDED:"
   - Extract and specify: table_name (ads or feeds), n_rows (number of rows to generate)
   - Format: "SYNTHETIC_DATA_NEEDED: table_name=ads, n_rows=50"
   - If table name is not clear, default to "ads"
   - If number of rows is not specified, use a reasonable default (50-100)

3. If SQL_NEEDED:
   - Generate MySQL SQL query(ies) to answer the question
   - For recommendation queries: Find items similar to what the user has interacted with (same task_id, advertiser_id, app_id, etc.)
   - If multiple queries are needed, provide them separated by semicolons or newlines
   - Only return the SQL query(ies), without any additional explanation or markdown formatting
   - Do not include code blocks (```sql or ```) around the query
   - Start your response with "SQL_NEEDED:" followed by the query(ies)
   - **CRITICAL FOR SAMPLING**: If the user asks for a random sample to perform calculations on (e.g., "sample 100 rows and calculate mean"), you MUST use a subquery. 
     INCORRECT: SELECT AVG(col) FROM table ORDER BY RAND() LIMIT 100 (aggregates whole table first)
     CORRECT: SELECT AVG(col) FROM (SELECT col FROM table ORDER BY RAND() LIMIT 100) as sample

4. If NO_SQL_NEEDED:
   - Provide a clear, helpful answer based on the schema and documentation
   - Start your response with "NO_SQL_NEEDED:" followed by your answer

Remember: Only use SQL when absolutely necessary. If the question can be answered from the schema or documentation alone, provide a direct answer instead.{guidance_section}"""
    
    return prompt

def create_short_sql_prompt(user_query, rag_context="", extra_instructions=""):
    """Create a short prompt for subsequent messages in a chat session (context already exists)."""
    rag_section = ""
    if rag_context:
        rag_section = f"\nAdditional Context from Documentation:\n{rag_context}\n"
    guidance_section = ""
    if extra_instructions:
        guidance_section = f"\nAdditional Guidance:\n{extra_instructions}\n"
    
    prompt = f"""User Question: {user_query}{rag_section}{guidance_section}

Follow the same process as before: 
- If explicitly requesting to GENERATE/CREATE synthetic data (with action verbs), respond with "SYNTHETIC_DATA_NEEDED: table_name=X, n_rows=Y"
- If asking ABOUT synthetic data generation (questions like "what process", "how does it work"), respond with "NO_SQL_NEEDED:" and explain the process
- If SQL is needed, respond with "SQL_NEEDED:" followed by the query(ies)
  * REMINDER: For sampling tasks ("sample X rows then..."), use a subquery: SELECT ... FROM (SELECT ... LIMIT X) as t
- If no SQL needed, respond with "NO_SQL_NEEDED:" followed by your answer

Use the database schema and context from our previous conversation."""
    
    return prompt

def extract_sql_from_response(response_text):
    """
    Extract SQL queries, synthetic data request, or direct answer from Gemini's response.
    
    Returns:
        Tuple (response_type, sql_queries_list, direct_answer, synthetic_params, chart_config)
        response_type: 'sql', 'synthetic', 'direct', or 'unknown'
    """
    response_text = response_text.strip()
    
    # Check for synthetic data request
    if response_text.startswith("SYNTHETIC_DATA_NEEDED:"):
        params_text = response_text.replace("SYNTHETIC_DATA_NEEDED:", "").strip()
        # Parse parameters: table_name=ads, n_rows=50
        table_match = re.search(r'table_name\s*=\s*(\w+)', params_text, re.IGNORECASE)
        rows_match = re.search(r'n_rows\s*=\s*(\d+)', params_text, re.IGNORECASE)
        
        table_name = table_match.group(1).lower() if table_match else 'ads'
        n_rows = int(rows_match.group(1)) if rows_match else 100
        
        # Validate table name
        if table_name not in ['ads', 'feeds']:
            table_name = 'ads'
        
        # Validate n_rows
        n_rows = min(max(n_rows, 1), 10000)
        
        return ('synthetic', [], None, {'table_name': table_name, 'n_rows': n_rows}, None)
    
    
    # Check if a chart is needed
    if "CHART_NEEDED:" in response_text:
        try:
            # Extract JSON from the CHART_NEEDED block
            chart_json_str = response_text.split("CHART_NEEDED:", 1)[1].strip()
            # If there is subsequent text, split it off (assuming chart block ends with strict JSON closing)
            # A simple heuristic: find the last '}'
            last_brace = chart_json_str.rfind('}')
            if last_brace != -1:
                chart_json_str = chart_json_str[:last_brace+1]
            
            # Clean up markdown code blocks if present in the chart JSON
            if chart_json_str.startswith("```"):
                lines = chart_json_str.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                chart_json_str = "\n".join(lines)
            
            chart_config = json.loads(chart_json_str)
            return ('direct', [], response_text.split("CHART_NEEDED:")[0].strip(), None, chart_config)
        except Exception as e:
            print(f"Error parsing chart JSON: {e}")
            # Fallback to direct text if chart parsing fails
            return ('direct', [], response_text.replace("CHART_NEEDED:", "").strip(), None, None)
    
    # Check if SQL is needed
    if response_text.startswith("NO_SQL_NEEDED:"):
        direct_answer = response_text.replace("NO_SQL_NEEDED:", "").strip()
        return ('direct', [], direct_answer, None, None)
    
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
    
    return ('sql', queries, None, None, None)

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

def dataframe_to_json(df):
    """Convert a DataFrame to a JSON-serializable format for frontend"""
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return {
                "columns": [],
                "data": [],
                "row_count": 0
            }
        # Convert DataFrame to dict with proper handling of NaN and other types
        # Use records orientation for easier frontend processing
        df_clean = df.fillna('')  # Replace NaN with empty strings
        return {
            "columns": df.columns.tolist(),
            "data": df_clean.to_dict('records'),  # List of dicts, one per row
            "row_count": len(df)
        }
    return None

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

def generate_combined_natural_language_response(
    user_query,
    all_sql_queries,
    all_results,
    rag_context="",
    chat_session=None,
    is_first_message=False,
    math_focus=False
):
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
    math_instruction = ""
    if math_focus:
        math_instruction = (
            "\nIMPORTANT: Perform any needed mathematical/statistical calculations "
            "using the data above (e.g., compute percentages, differences, test statistics). "
            "Show key numbers so the user can see how you derived the answer."
        )
    
    if is_first_message:
        prompt = f"""You are a data analyst assistant. A user asked a question about a database, and we executed SQL query(ies) to get the results. 

Original User Question: {user_query}
{rag_section}{sql_context}

Query Results:
{all_results_text}

Please provide a clear, concise, and easy-to-understand natural language explanation that answers the user's original question. Use the documentation context to provide richer explanations about variables, data structure, or domain knowledge when relevant. Combine all the results into one unified response. Keep the response brief but informative. If there are specific numbers, values, or patterns in the results, highlight them. If no results were returned, explain what that means in the context of the question. Focus on providing a single, cohesive answer that addresses the user's question directly.{math_instruction}

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

Provide a clear, concise natural language explanation answering the user's question. Use the same formatting guidelines as before.{math_instruction}"""
    
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

def generate_chart_config(user_query, all_results):
    """
    Generate a Chart.js configuration if the user requested a chart and data is available.
    Returns: JSON dict or None
    """
    # Quick check if a chart was requested
    chart_keywords = ["plot", "chart", "graph", "visualize", "visualization", "histogram", "scatter"]
    if not any(keyword in user_query.lower() for keyword in chart_keywords):
        return None
        
    # Format data for the prompt
    data_preview = ""
    if all_results:
        # Just use the first result that has data
        for sql, res in all_results:
            if isinstance(res, pd.DataFrame) and not res.empty:
                data_preview = res.head(20).to_string()
                break
    
    if not data_preview:
        return None

    prompt = f"""You are a data visualization assistant.
User Request: "{user_query}"
Data available (first 20 rows):
{data_preview}

Task: Create a Chart.js (version 3+) configuration JSON to visualize this data based on the user's request.
1. If the data is suitable for the requested chart (bar, line, pie, scatter, etc.), output ONLY the valid JSON object.
2. If the data cannot be visualized or no chart was explicitly requested, output "NO_CHART".

JSON Structure:
{{
  "type": "bar",
  "data": {{
    "labels": ["A", "B"],
    "datasets": [{{ "label": "Metric", "data": [10, 20], "backgroundColor": "rgba(75, 192, 192, 0.6)" }}]
  }},
  "options": {{ "plugins": {{ "title": {{ "display": true, "text": "Title" }} }} }}
}}

Return ONLY the raw JSON. No markdown, no backticks, no explanations."""

    try:
        gemini_model = get_model()
        # Use a low temperature for valid JSON
        response = gemini_model.generate_content(
            prompt, 
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        text = response.text.strip()
        
        if "NO_CHART" in text:
            return None
            
        # Cleanup markdown
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1] == "```": lines = lines[:-1]
            text = "\n".join(lines)
            
        return json.loads(text)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return None


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
        
        # Detect if extra math guidance is needed
        math_guidance = detect_math_intensive_request(user_query)
        
        # Retrieve relevant RAG context
        rag_context = ""
        if rag_collection:
            retrieved_chunks = retrieve_relevant_context(user_query, rag_collection, top_k=3)
            if retrieved_chunks:
                rag_context = format_rag_context(retrieved_chunks)
        
        # Create the prompt - full for first message, short for subsequent
        if is_first_message:
            prompt = create_full_sql_prompt(schema, user_query, rag_context, math_guidance or "")
        else:
            prompt = create_short_sql_prompt(user_query, rag_context, math_guidance or "")
        
        # Generate response from Gemini using chat session
        if is_first_message:
            # First message: send to chat session (this initializes the history)
            response = chat_session.send_message(prompt)
        else:
            # Subsequent messages: use chat session (maintains history)
            response = chat_session.send_message(prompt)
        
        # Extract response type and content
        response_type, sql_queries, direct_answer, synthetic_params, chart_config = extract_sql_from_response(response.text)
        
        # Fallback: Only use detection function if LLM response is unclear (not 'direct' or 'sql')
        # This prevents overriding correct answers to questions about the process
        if response_type not in ['synthetic', 'direct', 'sql']:
            detected = detect_synthetic_data_request(user_query)
            if detected:
                response_type = 'synthetic'
                synthetic_params = detected
        
        # Handle synthetic data generation request
        if response_type == 'synthetic':
            if SyntheticDataGenerator is None:
                return {
                    "success": False,
                    "error": "Synthetic data generator not available. Please check server logs.",
                    "answer": None,
                    "session_id": session_id
                }
            
            # Use parameters from LLM or fallback to detection
            if synthetic_params:
                table_name = synthetic_params['table_name']
                n_rows = synthetic_params['n_rows']
            else:
                # Fallback to detection function
                detected = detect_synthetic_data_request(user_query)
                if detected:
                    table_name = detected['table_name']
                    n_rows = detected['n_rows']
                else:
                    table_name = 'ads'
                    n_rows = 100
            
            try:
                # Get model and database connection
                gemini_model = get_model()
                connection = get_db_connection()
                
                # Initialize generator
                generator = SyntheticDataGenerator(
                    model=gemini_model,
                    schema=schema,
                    rag_collection=rag_collection,
                    db_connection=connection
                )
                
                # Generate synthetic data (RAG-heavy, token-efficient)
                result = generator.generate(
                    table_name=table_name,
                    n_rows=n_rows,
                    seed_size=50,  # Reduced for efficiency
                    use_rag=True,
                    validate=True,
                    use_llm_critique=False  # Skip for token savings
                )
                
                # Convert DataFrame to JSON-serializable format
                synthetic_df = result['data']
                table_json = dataframe_to_json(synthetic_df)
                
                # Generate natural language response
                stats_info = ""
                if result.get('statistics'):
                    stats = result['statistics']
                    if stats.get('correlation_similarity'):
                        stats_info = f" The synthetic data has a correlation similarity of {stats['correlation_similarity']:.2f} with the real data."
                
                answer = f"I've generated {result['row_count']} synthetic rows for the {table_name} table using a pure RAG approach.{stats_info} The data has been validated for schema compliance and statistical similarity."
                
                # Update chat metadata
                chat_metadata[session_id]["message_count"] += 1
                update_chat_title(session_id, user_query)
                
                return {
                    "success": True,
                    "answer": answer,
                    "sql_queries": [],
                    "needs_sql": False,
                    "is_synthetic": True,
                    "session_id": session_id,
                    "table_data": [table_json] if table_json else [],
                    "synthetic_stats": result.get('statistics')
                }
                
            except Exception as e:
                # Ensure connection is closed on error
                if connection:
                    try:
                        connection.close()
                    except:
                        pass
                return {
                    "success": False,
                    "error": f"Error generating synthetic data: {str(e)}",
                    "answer": None,
                    "session_id": session_id
                }
        
        # Handle case where SQL is not needed (direct answer)
        if response_type == 'direct':
            # Update chat metadata
            chat_metadata[session_id]["message_count"] += 1
            update_chat_title(session_id, user_query)
            
            return {
                "success": True,
                "answer": direct_answer,
                "sql_queries": [],
                "needs_sql": False,
                "session_id": session_id,
                "chart_config": chart_config
            }
        
        # Handle SQL queries
        if response_type != 'sql' or not sql_queries:
            return {
                "success": False,
                "error": "Could not determine request type or no queries found in the response.",
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
            # 1. Generate text response (using chat session context)
            natural_language_response = generate_combined_natural_language_response(
                user_query,
                sql_queries,
                all_results,
                rag_context,
                chat_session,
                is_first_message,
                math_focus=bool(math_guidance)
            )

            # 2. Generate chart config (independent step, fresh context)
            # This avoids polluting the chat history with JSON and ensures the model focuses only on the chart structure
            chart_config = generate_chart_config(user_query, all_results)
            
            final_answer = natural_language_response
            
            # Extract DataFrame results for table display/download
            table_data = []
            for sql_query, results in all_results:
                if isinstance(results, pd.DataFrame):
                    table_json = dataframe_to_json(results)
                    if table_json:
                        table_data.append(table_json)
            
            # Update chat metadata
            chat_metadata[session_id]["message_count"] += 1
            update_chat_title(session_id, user_query)
            
            response = {
                "success": True,
                "answer": final_answer,
                "sql_queries": executed_queries,
                "needs_sql": True,
                "session_id": session_id,
                "chart_config": chart_config
            }
            
            # Include table data if available (for CSV download and display)
            if table_data:
                response["table_data"] = table_data
            
            return response
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
            try:
                connection.close()
            except Exception:
                # Connection might already be closed, ignore
                pass

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
    import sys
    sys.stdout.flush()  # Ensure output is visible
    
    # Load schema
    try:
        schema = load_schema()
        print("✓ Schema loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠ Warning: Error loading schema: {e}")
        print("  App will continue but queries may fail until schema is available.")
        sys.stdout.flush()
        schema = None
    
    # Load RAG collection (non-blocking, can be slow)
    try:
        print("Loading RAG collection...")
        sys.stdout.flush()
        rag_collection = load_rag_collection()
        if rag_collection:
            chunk_count = rag_collection.count()
            print(f"✓ RAG system loaded ({chunk_count} chunks indexed)")
        else:
            print("⚠ RAG index not found. App will work without RAG context.")
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠ Warning: Could not load RAG system: {e}")
        print("  App will continue without RAG context.")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        rag_collection = None
    
    # Validate database connection (non-blocking)
    try:
        print("Testing database connection...")
        sys.stdout.flush()
        connection = get_db_connection()
        print("✓ Database connection established")
        connection.close()
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠ Warning: Could not connect to database: {e}")
        print("  App will start, but queries will fail until database is available.")
        sys.stdout.flush()
    
    # Check Gemini API key (non-blocking, won't initialize model yet)
    if not GOOGLE_API_KEY:
        print("⚠ Warning: GOOGLE_API_KEY not found in environment variables.")
        print("  App will start, but queries will fail until GOOGLE_API_KEY is set.")
    else:
        print("✓ GOOGLE_API_KEY found (model will be initialized on first use)")
    sys.stdout.flush()
    
    print("✓ Application initialization complete")
    sys.stdout.flush()

# Initialize on import (but don't crash if it fails)
# This runs when the module is imported by gunicorn
# Use threading to make it non-blocking so app can bind to port quickly
def initialize_async():
    """Initialize in background thread to avoid blocking port binding"""
    try:
        initialize()
    except Exception as e:
        print(f"⚠ Warning during initialization: {e}")
        print("  App will continue, but some features may not work.")
        import traceback
        traceback.print_exc()

# Start initialization in background thread
init_thread = threading.Thread(target=initialize_async, daemon=True)
init_thread.start()

# Health check endpoint for Render
@app.route('/health')
def health():
    """Health check endpoint for deployment platforms"""
    return jsonify({
        "status": "healthy", 
        "service": "database-chatbot",
        "schema_loaded": schema is not None,
        "rag_loaded": rag_collection is not None
    }), 200

if __name__ == '__main__':
    # For local development only (gunicorn ignores this)
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

