"""
Embedder module - generates vector embeddings for text chunks.
"""

from typing import List, Dict, Any
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self, embedder_type: str = "sentence-transformers"):
        """
        Initialize embedder.
        
        Args:
            embedder_type: Type of embedder (for compatibility)
        """
        self.embedder_type = embedder_type
        
        # Use HuggingFace embeddings (lightweight and fast)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk['text'] for chunk in chunks]
        return self.embed_texts(texts)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embeddings.embed_documents(texts)
        return [np.array(emb) for emb in embeddings]
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        embedding = self.embeddings.embed_query(query)
        return np.array(embedding)
