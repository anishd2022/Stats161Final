# Synthetic Data Generation - Hybrid Self-Instruct Approach

This module implements a hybrid self-instruct approach for generating high-quality synthetic data that combines:

1. **LLM-based generation** using Google Gemini
2. **Schema validation** to ensure data integrity
3. **Statistical validation** to maintain data distributions
4. **LLM self-critique** for quality control

## Features

- ✅ **Self-Instruct Pipeline**: LLM generates instructions and creates synthetic data
- ✅ **RAG Integration**: Uses existing RAG system for context-aware generation
- ✅ **Multi-level Validation**: Schema, statistical, and LLM-based quality checks
- ✅ **Scalable**: Generates data in chunks to avoid context limits
- ✅ **Flexible**: Supports both `ads` and `feeds` tables

## API Endpoints

### 1. Generate Synthetic Data

**POST** `/api/synthetic/generate`

Generate synthetic data for a specified table.

**Request Body:**
```json
{
  "table_name": "ads",  // or "feeds"
  "n_rows": 100,        // Target number of rows (default: 100, max: 10000)
  "seed_size": 100,     // Seed data sample size (default: 100, range: 10-1000)
  "n_instructions": 20, // Number of instructions to generate (default: 20)
  "rows_per_instruction": 5, // Rows per instruction (default: 5)
  "use_rag": true,      // Use RAG context (default: true)
  "validate": true      // Perform validation (default: true)
}
```

**Response:**
```json
{
  "success": true,
  "table_name": "ads",
  "row_count": 100,
  "data": {
    "columns": ["log_id", "label", "user_id", ...],
    "data": [{...}, {...}, ...],
    "row_count": 100
  },
  "statistics": {
    "distribution_similarity": {...},
    "correlation_similarity": 0.85,
    "class_balance": {...}
  },
  "instructions_generated": 20,
  "sample_instructions": [
    "Generate a record where the user is using a mobile device...",
    ...
  ]
}
```

### 2. Get Available Tables

**GET** `/api/synthetic/tables`

Get list of tables available for synthetic data generation.

**Response:**
```json
{
  "success": true,
  "tables": ["ads", "feeds"]
}
```

### 3. Get Statistical Comparison

**POST** `/api/synthetic/stats`

Compare synthetic data statistics with real data.

**Request Body:**
```json
{
  "table_name": "ads",
  "synthetic_data": [{...}, {...}, ...],
  "sample_size": 1000  // Size of real data sample (default: 1000)
}
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "distribution_similarity": {...},
    "correlation_similarity": 0.85,
    "class_balance": {...}
  },
  "synthetic_row_count": 100,
  "real_row_count": 1000
}
```

## Usage Examples

### Python Example

```python
import requests

# Generate 100 synthetic rows for ads table
response = requests.post('http://localhost:5001/api/synthetic/generate', json={
    'table_name': 'ads',
    'n_rows': 100,
    'seed_size': 100,
    'n_instructions': 20,
    'validate': True
})

result = response.json()
if result['success']:
    print(f"Generated {result['row_count']} rows")
    print(f"Statistics: {result['statistics']}")
```

### cURL Example

```bash
curl -X POST http://localhost:5001/api/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "table_name": "ads",
    "n_rows": 100,
    "validate": true
  }'
```

## How It Works

### 1. Seed Phase
- Samples diverse rows from the database (default: 100 rows)
- Extracts column information (types, ranges, distributions)

### 2. Instruction Generation
- LLM generates diverse instructions based on seed data and schema
- Uses RAG context for domain knowledge
- Creates instructions like:
  - "Generate a record where the user is using a mobile device and clicked on an ad"
  - "Generate a record with very low advertisement quality score"

### 3. Data Generation
- For each instruction, LLM generates multiple rows
- Ensures all columns are present and properly formatted
- Handles JSON columns, timestamps, and categorical data

### 4. Validation Pipeline
- **Schema Validation**: Checks data types, constraints, ranges
- **Statistical Validation**: Compares distributions, correlations, class balance
- **LLM Self-Critique**: LLM reviews and filters unrealistic rows

### 5. Quality Metrics
- Distribution similarity (Kolmogorov-Smirnov test)
- Correlation preservation
- Class balance maintenance
- Schema compliance rate

## Configuration

The generator uses the following defaults:
- **Seed size**: 100 rows (enough for pattern learning, small enough for context)
- **Instructions**: 20 (diverse scenarios)
- **Rows per instruction**: 5 (balance between diversity and quality)
- **Validation**: Enabled by default

## Best Practices

1. **Start Small**: Begin with 50-100 rows to test quality
2. **Use Validation**: Keep `validate: true` for production use
3. **Adjust Seed Size**: Larger seed sizes (200-500) for complex datasets
4. **Monitor Statistics**: Check `statistics` in response to ensure quality
5. **Iterate**: Use feedback to refine generation parameters

## Limitations

- **Context Limits**: Very large seed sizes may hit token limits
- **Statistical Fidelity**: LLM-generated data may not perfectly match distributions
- **Complex Relationships**: Cross-table relationships require careful handling
- **JSON Columns**: Complex nested JSON may need manual validation

## Troubleshooting

**Error: "Synthetic data generator not available"**
- Check that `synthetic_generator.py` is in the same directory as `app.py`
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Error: "Failed to parse JSON from LLM response"**
- LLM sometimes returns formatted text instead of pure JSON
- The generator attempts to extract JSON, but may fail occasionally
- Try reducing `rows_per_instruction` or `n_instructions`

**Low Quality Data**
- Increase `seed_size` for better pattern learning
- Enable `validate: true` for quality filtering
- Check `statistics` in response to identify issues
- Adjust `n_instructions` for more diverse scenarios

## Integration with Existing System

The synthetic data generator integrates seamlessly with:
- ✅ Existing RAG system for context
- ✅ Database connection for seed data
- ✅ Schema information from `schema_clean.sql`
- ✅ Variable descriptions from RAG docs

## Future Enhancements

- [ ] Support for cross-table generation (ads + feeds)
- [ ] Advanced statistical validation (copula-based)
- [ ] Fine-tuning based on quality metrics
- [ ] Batch generation with progress tracking
- [ ] Export to CSV/SQL format

