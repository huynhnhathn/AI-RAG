"""
Qdrant Vector Store Configuration and Management
"""
import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class QdrantVectorStore:
    """Manages Qdrant vector database operations for the AI assistant."""
    
    def __init__(self, 
                 collection_name: str = None,
                 qdrant_url: str = None,
                 qdrant_api_key: str = None):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of Qdrant instance
            qdrant_api_key: API key for Qdrant (if required)
        """
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "ai_assistant_docs")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize vector store
        self.vector_store = None
        self._setup_collection()
    
    def _setup_collection(self):
        """Set up the Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with appropriate vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created new collection: {self.collection_name}")
            else:
                print(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error setting up collection: {e}")
            raise
    
    def get_vector_store(self) -> Qdrant:
        """Get or create the Langchain Qdrant vector store instance."""
        if self.vector_store is None:
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        vector_store = self.get_vector_store()
        return vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of (document, score) tuples
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search_with_score(query, k=k)
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self):
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None