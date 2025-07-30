"""
Retrieval Chain Implementation
"""
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.retrievers import VectorStoreRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from .vector_store import QdrantVectorStore


class AIRetriever:
    """Advanced retrieval system for the AI assistant."""
    
    def __init__(self, 
                 vector_store: QdrantVectorStore,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 top_k: int = 4,
                 use_multi_query: bool = False,
                 use_compression: bool = False):
        """
        Initialize the retriever.
        
        Args:
            vector_store: QdrantVectorStore instance
            model_name: OpenAI model name
            temperature: Model temperature
            top_k: Number of documents to retrieve
            use_multi_query: Whether to use multi-query retrieval
            use_compression: Whether to use contextual compression
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize base retriever
        self.base_retriever = self._create_base_retriever()
        
        # Initialize advanced retrievers
        self.multi_query_retriever = None
        self.compression_retriever = None
        
        if use_multi_query:
            self.multi_query_retriever = self._create_multi_query_retriever()
        
        if use_compression:
            self.compression_retriever = self._create_compression_retriever()
        
        # Initialize memory for conversational retrieval
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def _create_base_retriever(self) -> VectorStoreRetriever:
        """Create the base vector store retriever."""
        langchain_vector_store = self.vector_store.get_vector_store()
        return VectorStoreRetriever(
            vectorstore=langchain_vector_store,
            search_kwargs={"k": self.top_k}
        )
    
    def _create_multi_query_retriever(self) -> MultiQueryRetriever:
        """Create multi-query retriever for better results."""
        return MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
    
    def _create_compression_retriever(self) -> ContextualCompressionRetriever:
        """Create compression retriever to filter relevant content."""
        compressor = LLMChainExtractor.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def retrieve_documents(self, 
                          query: str, 
                          retriever_type: str = "base") -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            retriever_type: Type of retriever ("base", "multi_query", "compression")
            
        Returns:
            List of relevant documents
        """
        try:
            if retriever_type == "multi_query" and self.multi_query_retriever:
                return self.multi_query_retriever.get_relevant_documents(query)
            elif retriever_type == "compression" and self.compression_retriever:
                return self.compression_retriever.get_relevant_documents(query)
            else:
                return self.base_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def create_qa_chain(self, 
                       chain_type: str = "stuff",
                       custom_prompt: str = None) -> RetrievalQA:
        """
        Create a question-answering chain.
        
        Args:
            chain_type: Type of QA chain ("stuff", "map_reduce", "refine", "map_rerank")
            custom_prompt: Custom prompt template
            
        Returns:
            RetrievalQA chain
        """
        # Default prompt template
        default_prompt = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Answer:"""
        
        if custom_prompt:
            prompt_template = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question"]
            )
        else:
            prompt_template = PromptTemplate(
                template=default_prompt,
                input_variables=["context", "question"]
            )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.base_retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
    
    def create_conversational_chain(self, 
                                  custom_prompt: str = None) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain with memory.
        
        Args:
            custom_prompt: Custom prompt template
            
        Returns:
            ConversationalRetrievalChain
        """
        # Default conversational prompt
        default_prompt = """Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        
        if custom_prompt:
            condense_question_prompt = PromptTemplate.from_template(custom_prompt)
        else:
            condense_question_prompt = PromptTemplate.from_template(default_prompt)
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.base_retriever,
            memory=self.memory,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=True
        )
    
    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Search for documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_relevant_context(self, 
                           query: str, 
                           max_context_length: int = 4000) -> str:
        """
        Get relevant context for a query, truncated to max length.
        
        Args:
            query: Search query
            max_context_length: Maximum context length in characters
            
        Returns:
            Concatenated context string
        """
        documents = self.retrieve_documents(query)
        
        context_parts = []
        current_length = 0
        
        for doc in documents:
            content = doc.page_content
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Add partial content if it fits
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if significant space remains
                    context_parts.append(content[:remaining_space] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    def update_retriever_config(self, **kwargs):
        """
        Update retriever configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if 'top_k' in kwargs:
            self.top_k = kwargs['top_k']
            self.base_retriever.search_kwargs = {"k": self.top_k}
        
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
            self.llm.temperature = self.temperature
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever and vector store.
        
        Returns:
            Dictionary with retriever statistics
        """
        collection_info = self.vector_store.get_collection_info()
        
        stats = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "has_multi_query": self.multi_query_retriever is not None,
            "has_compression": self.compression_retriever is not None,
            "collection_name": self.vector_store.collection_name,
        }
        
        if collection_info:
            stats.update({
                "total_documents": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value
            })
        
        return stats