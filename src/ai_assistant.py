"""
Main AI Assistant Class
"""
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.documents import Document

from .vector_store import QdrantVectorStore
from .document_loader import DocumentProcessor
from .retriever import AIRetriever

# Load environment variables
load_dotenv()


class AIAssistant:
    """
    Main AI Assistant class that combines vector storage, document processing,
    and retrieval-augmented generation capabilities.
    """
    
    def __init__(self,
                 collection_name: str = None,
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 openai_api_key: str = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k: int = 4,
                 use_multi_query: bool = False,
                 use_compression: bool = False):
        """
        Initialize the AI Assistant.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of Qdrant instance  
            qdrant_api_key: API key for Qdrant
            openai_api_key: OpenAI API key
            model_name: OpenAI model name
            temperature: Model temperature
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            top_k: Number of documents to retrieve
            use_multi_query: Whether to use multi-query retrieval
            use_compression: Whether to use contextual compression
        """
        # Set API keys
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key must be provided or set in environment")
        
        # Initialize components
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.retriever = AIRetriever(
            vector_store=self.vector_store,
            model_name=model_name,
            temperature=temperature,
            top_k=top_k,
            use_multi_query=use_multi_query,
            use_compression=use_compression
        )
        
        # Initialize chains
        self.qa_chain = None
        self.conversational_chain = None
        
        print(f"AI Assistant initialized with model: {model_name}")
        print(f"Vector store collection: {self.vector_store.collection_name}")
    
    def add_documents_from_files(self, 
                               file_paths: Union[str, List[str]],
                               chunk: bool = True) -> List[str]:
        """
        Add documents from files to the knowledge base.
        
        Args:
            file_paths: Single file path or list of file paths
            chunk: Whether to chunk the documents
            
        Returns:
            List of document IDs
        """
        print(f"Processing documents from files: {file_paths}")
        
        # Process documents
        documents = self.document_processor.process_documents(
            file_paths=file_paths,
            chunk=chunk
        )
        
        if not documents:
            print("No documents were processed")
            return []
        
        print(f"Processed {len(documents)} document chunks")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        print(f"Added {len(doc_ids)} documents to vector store")
        
        return doc_ids
    
    def add_documents_from_directory(self, 
                                   directory_path: str,
                                   chunk: bool = True) -> List[str]:
        """
        Add documents from a directory to the knowledge base.
        
        Args:
            directory_path: Path to directory containing documents
            chunk: Whether to chunk the documents
            
        Returns:
            List of document IDs
        """
        print(f"Processing documents from directory: {directory_path}")
        
        # Process documents
        documents = self.document_processor.process_documents(
            directory_path=directory_path,
            chunk=chunk
        )
        
        if not documents:
            print("No documents were processed")
            return []
        
        print(f"Processed {len(documents)} document chunks")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        print(f"Added {len(doc_ids)} documents to vector store")
        
        return doc_ids
    
    def add_text_content(self, 
                        text: str, 
                        metadata: Dict[str, Any] = None,
                        chunk: bool = True) -> List[str]:
        """
        Add text content directly to the knowledge base.
        
        Args:
            text: Text content to add
            metadata: Optional metadata for the text
            chunk: Whether to chunk the text
            
        Returns:
            List of document IDs
        """
        print("Processing text content")
        
        # Process text
        documents = self.document_processor.process_documents(
            text_content=text,
            chunk=chunk
        )
        
        # Add metadata if provided
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        if not documents:
            print("No documents were processed")
            return []
        
        print(f"Processed {len(documents)} document chunks")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        print(f"Added {len(doc_ids)} documents to vector store")
        
        return doc_ids
    
    def ask_question(self, 
                    question: str, 
                    return_sources: bool = True,
                    retriever_type: str = "base") -> Dict[str, Any]:
        """
        Ask a question and get an answer with retrieved context.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            retriever_type: Type of retriever to use
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        try:
            # Initialize QA chain if not exists
            if self.qa_chain is None:
                self.qa_chain = self.retriever.create_qa_chain()
            
            # Get answer
            response = self.qa_chain({"query": question})
            
            result = {
                "question": question,
                "answer": response["result"],
            }
            
            if return_sources and "source_documents" in response:
                result["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in response["source_documents"]
                ]
            
            return result
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": "I'm sorry, I encountered an error while processing your question.",
                "error": str(e)
            }
    
    def chat(self, 
             message: str, 
             return_sources: bool = True) -> Dict[str, Any]:
        """
        Have a conversational chat with memory.
        
        Args:
            message: Message to send
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with response and optionally source documents
        """
        try:
            # Initialize conversational chain if not exists
            if self.conversational_chain is None:
                self.conversational_chain = self.retriever.create_conversational_chain()
            
            # Get response
            response = self.conversational_chain({"question": message})
            
            result = {
                "message": message,
                "response": response["answer"],
            }
            
            if return_sources and "source_documents" in response:
                result["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in response["source_documents"]
                ]
            
            return result
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return {
                "message": message,
                "response": "I'm sorry, I encountered an error while processing your message.",
                "error": str(e)
            }
    
    def search_documents(self, 
                        query: str, 
                        k: int = None,
                        with_scores: bool = False) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            with_scores: Whether to include relevance scores
            
        Returns:
            List of documents with metadata and optionally scores
        """
        try:
            if with_scores:
                results = self.retriever.search_with_scores(query, k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                ]
            else:
                documents = self.retriever.retrieve_documents(query)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the AI assistant.
        
        Returns:
            Dictionary with assistant statistics
        """
        try:
            retriever_stats = self.retriever.get_retriever_stats()
            collection_info = self.vector_store.get_collection_info()
            
            stats = {
                "assistant_config": {
                    "model_name": self.retriever.model_name,
                    "temperature": self.retriever.temperature,
                    "chunk_size": self.document_processor.chunk_size,
                    "chunk_overlap": self.document_processor.chunk_overlap,
                },
                "retriever_stats": retriever_stats,
                "vector_store": {
                    "collection_name": self.vector_store.collection_name,
                    "qdrant_url": self.vector_store.qdrant_url,
                }
            }
            
            if collection_info:
                stats["vector_store"]["document_count"] = collection_info.points_count
                stats["vector_store"]["vector_size"] = collection_info.config.params.vectors.size
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def clear_memory(self):
        """Clear conversational memory."""
        if self.retriever.memory:
            self.retriever.memory.clear()
            print("Conversational memory cleared")
    
    def delete_collection(self):
        """Delete the entire vector collection."""
        self.vector_store.delete_collection()
        print("Vector collection deleted")
    
    def update_config(self, **kwargs):
        """
        Update assistant configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        # Update retriever config
        retriever_params = {k: v for k, v in kwargs.items() 
                          if k in ['top_k', 'temperature']}
        if retriever_params:
            self.retriever.update_retriever_config(**retriever_params)
        
        # Update document processor config
        if 'chunk_size' in kwargs:
            self.document_processor.chunk_size = kwargs['chunk_size']
        if 'chunk_overlap' in kwargs:
            self.document_processor.chunk_overlap = kwargs['chunk_overlap']
        
        # Reinitialize text splitter if chunk parameters changed
        if 'chunk_size' in kwargs or 'chunk_overlap' in kwargs:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.document_processor.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.document_processor.chunk_size,
                chunk_overlap=self.document_processor.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
        print("Configuration updated")