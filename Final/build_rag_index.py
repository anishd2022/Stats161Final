"""
Build RAG Index - Run this once to create embeddings from documents
Run this script whenever you add or update documents in rag_docs/
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# Get the directory where this script is located
script_dir = Path(__file__).parent
rag_docs_dir = script_dir / "rag_docs"
chroma_db_path = script_dir / "chroma_db"

# Load environment variables
env_path = script_dir / ".env"
load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. "
        "Please ensure your .env file contains GOOGLE_API_KEY variable."
    )

def load_documents():
    """Load all text documents from the rag_docs folder."""
    documents = []
    
    if not rag_docs_dir.exists():
        print(f"Error: RAG docs directory not found: {rag_docs_dir}")
        return documents
    
    supported_extensions = {'.txt', '.md', '.markdown'}
    
    for file_path in rag_docs_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append((file_path.name, content))
                print(f"  ✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"  ✗ Error loading {file_path.name}: {e}")
    
    return documents

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks with overlap. Optimized for speed."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    # Calculate expected number of chunks for safety limit
    expected_chunks = (text_len // (chunk_size - chunk_overlap)) + 2
    max_iterations = expected_chunks * 2  # Safety limit (2x expected)
    iteration = 0
    
    while start < text_len and iteration < max_iterations:
        iteration += 1
        end = min(start + chunk_size, text_len)
        
        # Quick boundary detection - only check last 200 chars for performance
        if end < text_len:
            search_start = max(start, end - 200)
            # Look for paragraph break first (fastest)
            para_pos = text.rfind('\n\n', search_start, end)
            if para_pos != -1:
                end = para_pos + 2
            else:
                # Look for sentence breaks
                period_pos = text.rfind('. ', search_start, end)
                if period_pos != -1:
                    end = period_pos + 2
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        new_start = max(end - chunk_overlap, start + 1)  # Always advance by at least 1
        if new_start >= text_len:
            break
        start = new_start
    
    return chunks

def build_rag_index():
    """Build the RAG index by chunking documents and creating embeddings."""
    print("="*80)
    print("Building RAG Index")
    print("="*80)
    
    # Load documents
    print("\nLoading documents from rag_docs/...")
    documents = load_documents()
    
    if not documents:
        print("No documents found. Nothing to index.")
        return
    
    print(f"\nLoaded {len(documents)} document(s)")
    
    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(chroma_db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = "rag_documents"
    
    # Delete existing collection if it exists (to rebuild from scratch)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection with sentence-transformers embedding
    # This avoids ONNX runtime compatibility issues
    print("Creating collection with sentence-transformers embeddings...")
    print("(First run will download the model, ~80MB - this is a one-time download)")
    try:
        from chromadb.utils import embedding_functions
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"✓ Created collection: {collection_name} (using sentence-transformers)")
    except Exception as e:
        print(f"✗ Error creating collection with sentence-transformers: {e}")
        print("Falling back to simple text-based approach...")
        # Fallback: use a simple embedding function
        raise ValueError(
            "Could not create embeddings. Please ensure sentence-transformers is properly installed.\n"
            "Try: pip install --upgrade sentence-transformers"
        )
    
    # Chunk documents
    print("\nChunking documents...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for doc_idx, (filename, content) in enumerate(documents, 1):
        print(f"  Processing {filename} ({doc_idx}/{len(documents)})...", end=' ', flush=True)
        chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
        print(f"{len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            all_ids.append(f"{filename}_{i}")
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    
    # Create embeddings and add to ChromaDB in batches to avoid memory issues
    print("\nCreating embeddings and adding to index (this may take a minute)...")
    print("Note: ChromaDB uses its default embedding model for fast indexing")
    
    # Add to ChromaDB in batches to avoid memory issues
    # Use smaller batch size to reduce memory pressure
    batch_size = 50  # Process 50 chunks at a time
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    import gc  # For garbage collection
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_chunks))
        
        batch_chunks = all_chunks[start_idx:end_idx]
        batch_metadatas = all_metadatas[start_idx:end_idx]
        batch_ids = all_ids[start_idx:end_idx]
        
        try:
            collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"  Processed batch {batch_num + 1}/{total_batches} ({end_idx}/{len(all_chunks)} chunks)")
            
            # Clear batch data and force garbage collection to free memory
            del batch_chunks, batch_metadatas, batch_ids
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Error processing batch {batch_num + 1}: {e}")
            raise
    
    print(f"\n✓ Successfully indexed {len(all_chunks)} chunks from {len(documents)} documents")
    print(f"✓ Index stored in: {chroma_db_path}")
    print("\n" + "="*80)
    print("RAG Index Build Complete!")
    print("="*80)
    print("\nYou can now run query.py to use the RAG system.")
    print("Re-run this script if you add or update documents in rag_docs/")

if __name__ == "__main__":
    try:
        build_rag_index()
    except Exception as e:
        print(f"\n✗ Error building RAG index: {e}")
        raise

