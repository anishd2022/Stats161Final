# Database Query Chatbot

A natural language interface for querying a MySQL database using Google Gemini AI. The system uses RAG (Retrieval-Augmented Generation) to provide context-aware responses and can answer questions about the database schema, variable definitions, and execute SQL queries based on natural language questions.

## Features

- **Natural Language to SQL**: Convert questions in plain English to SQL queries
- **RAG-Enhanced Context**: Uses document retrieval to provide better context about variables and schema
- **Web Interface**: Flask-based web application with a chat interface
- **Command-Line Interface**: Interactive CLI for querying the database
- **Smart Query Detection**: Automatically determines if a question needs SQL or can be answered from documentation

## Prerequisites

- Python 3.11 or higher
- MySQL database server (running and accessible)
- Google Gemini API key
- Virtual environment (recommended)

## Installation

1. **Navigate to the Final directory:**
   ```bash
   cd Final
   ```

2. **Create and activate a virtual environment (if not already done):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Create a `.env` file in the `Final/` directory with the following variables:

```env
# Google Gemini API Key (required)
GOOGLE_API_KEY=your_google_api_key_here

# MySQL Database Configuration (required)
DB_HOST=localhost
DB_USER=your_database_user
DB_PW=your_database_password
DB_NAME=your_database_name
DB_PORT=3306
```

**Important:** 
- Replace all placeholder values with your actual credentials
- Never commit the `.env` file to version control
- The `.env` file should be in the same directory as `app.py` and `query.py`

### Database Setup

The project includes multiple schema files:
- `schema_clean.sql` - Main database schema for ads and feeds tables
- `schema.sql` - Alternative schema definition
- `schema_generative.sql` - Schema for generative/synthetic data tables

Ensure your MySQL database is set up with one of the schema files. The application expects tables named `ads` and `feeds` with the structure defined in the schema file.

#### Populating the Database

After setting up the schema, you can populate the database with data:

**For regular training data:**
```bash
python populate_db.py
```
This script loads data from `train_data_ads.csv` and `train_data_feeds.csv` into the database.

**For synthetic data (SMOTE/ADASYN):**
```bash
python populate_synthetic_db.py
```
This script loads synthetic data from `synthetic_train_SMOTE_raw.csv` and `synthetic_train_ADASYN_raw.csv` into the database.

## Building the RAG Index

Before using the system, you need to build the RAG index from the documentation files:

```bash
python build_rag_index.py
```

This script will:
- Load all text files from the `rag_docs/` directory
- Chunk the documents into smaller pieces
- Create embeddings using sentence-transformers
- Store the index in `chroma_db/`

**Note:** 
- The first run will download the embedding model (~80MB)
- Re-run this script whenever you update documents in `rag_docs/`
- The RAG system is optional - the application will work without it, but with less context

The RAG documentation includes information about:
- Dataset overview and variable descriptions
- Database schema details
- Synthetic data generation methods (SMOTE, ADASYN)
- Guidelines for generating synthetic data

## Running the Application

### Web Interface (Flask App)

Start the Flask web application:

```bash
python app.py
```

The application will start on `http://0.0.0.0:5001` (or `http://localhost:5001`).

Open your web browser and navigate to:
```
http://localhost:5001
```

You'll see a chat interface where you can ask questions about the database.

### Command-Line Interface

For an interactive command-line experience:

```bash
python query.py
```

This will start an interactive session where you can:
- Type questions about the database
- See generated SQL queries
- View query results
- Get natural language explanations

Type `quit` or `exit` to end the session.

## Project Structure

```
Final/
├── app.py                      # Flask web application
├── query.py                    # Command-line interface
├── rag_system.py               # RAG system implementation
├── build_rag_index.py          # Script to build RAG index
├── populate_db.py              # Script to populate database with training data
├── populate_synthetic_db.py    # Script to populate database with synthetic data
├── synthetic_generator.py      # Synthetic data generation using RAG
├── analyze_data_statistics.py  # Script to analyze database statistics
├── requirements.txt            # Python dependencies
├── schema_clean.sql            # Main database schema definition
├── schema.sql                  # Alternative schema definition
├── schema_generative.sql       # Schema for synthetic data tables
├── DEPLOYMENT.md               # Deployment guide for hosting the application
├── Procfile                    # Process file for deployment (Heroku/Render)
├── runtime.txt                 # Python version specification
├── start.sh                    # Startup script for deployment
├── .env                        # Environment variables (create this)
├── chroma_db/                  # RAG index storage (created by build_rag_index.py)
├── rag_docs/                   # Documentation files for RAG
│   ├── about_dataset.txt
│   ├── ads_variable_descriptions.txt
│   ├── database_schema.txt
│   ├── feeds_variable_descriptions.txt
│   ├── synthetic_data_generation_guide.txt
│   └── synthetic_smote_adasyn_datasets.txt
├── templates/                  # HTML templates for web app
│   └── index.html
├── static/                     # CSS and JavaScript for web app
│   ├── style.css
│   └── script.js
├── train_data_ads.csv          # Training data for ads table
├── train_data_feeds.csv        # Training data for feeds table
├── synthetic_train_SMOTE_raw.csv    # Synthetic data generated using SMOTE
├── synthetic_train_ADASYN_raw.csv   # Synthetic data generated using ADASYN
└── sample_questions.txt        # Example questions you can try
```

## Usage Examples

### Example Questions

The system can handle various types of questions:

**Schema/Documentation Questions (no SQL needed):**
- "What is the user_id field in the ads table?"
- "How are the ads and feeds tables related?"
- "What does the label column represent?"

**Data Query Questions (SQL needed):**
- "How many users are in the database?"
- "What is the average age of users who clicked on ads?"
- "Show me the top 10 advertisers by number of ads"

### Sample Questions File

Check `sample_questions.txt` for more example questions you can try.

## Troubleshooting

### Common Issues

1. **"GOOGLE_API_KEY not found"**
   - Ensure your `.env` file exists in the `Final/` directory
   - Check that `GOOGLE_API_KEY` is set correctly

2. **"Database connection failed"**
   - Verify your MySQL server is running
   - Check database credentials in `.env`
   - Ensure the database exists and schema is loaded

3. **"RAG index not found"**
   - Run `python build_rag_index.py` to create the index
   - The application will work without RAG, but with less context

4. **"Model not found" errors**
   - The code uses `gemini-2.5-flash` by default
   - If unavailable, check available models and update `MODEL_NAME` in the code

5. **Import errors**
   - Ensure you're in the virtual environment
   - Run `pip install -r requirements.txt` again

## Dependencies

Key dependencies include:
- `flask` - Web framework
- `google-generativeai` - Google Gemini API client
- `pymysql` - MySQL database connector
- `pandas` - Data manipulation
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Embeddings for RAG
- `python-dotenv` - Environment variable management

See `requirements.txt` for the complete list.

## Additional Tools

### Synthetic Data Generation

The project includes a synthetic data generator (`synthetic_generator.py`) that uses RAG to generate synthetic data following the patterns and statistical properties of the original dataset. This can be used to augment training data or create test datasets.

### Data Analysis

The `analyze_data_statistics.py` script analyzes the database to extract statistical properties, correlations, and distributions. This information is used to improve synthetic data generation and provide insights about the dataset.

## Deployment

For deployment instructions, see `DEPLOYMENT.md`. The project includes configuration files for deployment platforms:
- `Procfile` - Process configuration for Heroku/Render
- `runtime.txt` - Python version specification
- `start.sh` - Startup script

## Notes

- The application uses Google Gemini 2.5 Flash model by default
- SQL queries are automatically generated and executed
- Results are limited to 100 rows by default to avoid token limits
- The RAG system retrieves top 3 relevant document chunks per query
- All database queries use parameterized queries for security
- The system supports both original training data and synthetic data (SMOTE/ADASYN)

## License

This project is part of a course assignment (STATS C161).

