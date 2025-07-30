#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all imports from the AI assistant modules."""
    print("Testing imports...")
    
    try:
        # Test core langchain imports
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import Qdrant
        print("✓ Core LangChain imports successful")
        
        # Test Qdrant client
        from qdrant_client import QdrantClient
        print("✓ Qdrant client import successful")
        
        # Test our modules
        from src.vector_store import QdrantVectorStore
        print("✓ Vector store module import successful")
        
        from src.document_loader import DocumentProcessor
        print("✓ Document loader module import successful")
        
        from src.retriever import AIRetriever
        print("✓ Retriever module import successful")
        
        from src.ai_assistant import AIAssistant
        print("✓ AI assistant module import successful")
        
        print("\n🎉 All imports successful! The project is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)