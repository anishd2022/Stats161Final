"""
RAG (Retrieval-Augmented Generation) System for Query Enhancement
Handles document loading, chunking, embedding, and retrieval
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings

# Get the directory where this script is located
script_dir = Path(__file__).parent
rag_docs_dir = script_dir / "rag_docs"
chroma_db_path = script_dir / "chroma_db"

def load_documents():
    """
    Load all text documents from the rag_docs folder.
    
    Returns:
        List of tuples (filename, content)
    """
    documents = []
    
    if not rag_docs_dir.exists():
        print(f"Warning: RAG docs directory not found: {rag_docs_dir}")
        return documents
    
    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.markdown'}
    
    for file_path in rag_docs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append((file_path.name, content))
                print(f"  Loaded: {file_path.name}")
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
    
    return documents

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk
        end = start + chunk_size
        
        # Try to break at sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break (double newline)
            para_break = text.rfind('\n\n', start, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Look for sentence break (period, exclamation, question mark)
                sentence_breaks = [text.rfind('. ', start, end),
                                 text.rfind('! ', start, end),
                                 text.rfind('? ', start, end)]
                sentence_breaks = [b for b in sentence_breaks if b != -1]
                if sentence_breaks:
                    end = max(sentence_breaks) + 2
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start < 0:
            start = end
    
    return chunks

def load_rag_collection():
    """
    Load the existing RAG collection (assumes index has already been built).
    Run build_rag_index.py first if the index doesn't exist.
    
    Returns:
        ChromaDB collection with embedded documents, or None if not found
    """
    try:
        # Initialize ChromaDB
        client = chromadb.PersistentClient(
            path=str(chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get existing collection (using sentence-transformers embeddings)
        collection_name = "rag_documents"
        try:
            from chromadb.utils import embedding_functions
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
        except Exception as e:
            # Try without embedding function (for backwards compatibility)
            collection = client.get_collection(name=collection_name)
        
        chunk_count = collection.count()
        if chunk_count > 0:
            return collection
        else:
            print("Warning: RAG collection exists but is empty.")
            print("Run build_rag_index.py to create the index.")
            return None
            
    except Exception as e:
        # Collection doesn't exist
        return None

def retrieve_relevant_context(query, collection, top_k=3):
    """
    Retrieve relevant document chunks based on the query.
    
    Args:
        query: User's query string
        collection: ChromaDB collection
        top_k: Number of top results to retrieve
    
    Returns:
        List of relevant text chunks with metadata
    """
    if collection is None:
        return []
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                retrieved_chunks.append({
                    'text': doc,
                    'source': metadata.get('source', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', 0)
                })
        
        return retrieved_chunks
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def format_rag_context(retrieved_chunks):
    """
    Format retrieved chunks into a readable context string.
    
    Args:
        retrieved_chunks: List of chunk dictionaries
    
    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return ""
    
    context_parts = []
    context_parts.append("Relevant Context from Documentation:")
    context_parts.append("-" * 80)
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"\n[Context {i} - Source: {chunk['source']}]")
        context_parts.append(chunk['text'])
        context_parts.append("")
    
    return "\n".join(context_parts)

