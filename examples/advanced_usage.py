"""
Advanced Usage Example for AI Assistant
Demonstrates loading documents from files and directories
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_assistant import AIAssistant


def create_sample_documents():
    """Create sample documents for demonstration."""
    docs_dir = Path("sample_docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    files_created = []
    
    # Python documentation
    python_doc = docs_dir / "python_guide.txt"
    with open(python_doc, "w") as f:
        f.write("""
Python Programming Guide

Python is a versatile programming language that's great for beginners and experts alike.

Key Features:
- Easy to read and write syntax
- Extensive standard library
- Cross-platform compatibility
- Strong community support
- Object-oriented and functional programming support

Common Use Cases:
- Web development with frameworks like Django and Flask
- Data science and machine learning with pandas, numpy, scikit-learn
- Automation and scripting
- Desktop applications with tkinter or PyQt
- API development with FastAPI

Getting Started:
1. Install Python from python.org
2. Set up a virtual environment
3. Start with basic syntax and data types
4. Practice with small projects
        """)
    files_created.append(str(python_doc))
    
    # AI/ML documentation
    ai_doc = docs_dir / "ai_ml_guide.md"
    with open(ai_doc, "w") as f:
        f.write("""
# Artificial Intelligence and Machine Learning Guide

## What is AI?
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.

## Machine Learning Basics
Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.

### Types of Machine Learning:
1. **Supervised Learning**: Learning with labeled data
   - Classification (predicting categories)
   - Regression (predicting continuous values)

2. **Unsupervised Learning**: Learning patterns from unlabeled data
   - Clustering
   - Dimensionality reduction

3. **Reinforcement Learning**: Learning through interaction with environment
   - Agent learns through rewards and penalties

### Popular ML Libraries:
- **Scikit-learn**: General-purpose ML library
- **TensorFlow**: Deep learning framework by Google
- **PyTorch**: Deep learning framework by Facebook
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## Deep Learning
Deep Learning is a subset of ML that uses neural networks with multiple layers.

### Applications:
- Image recognition
- Natural language processing
- Speech recognition
- Autonomous vehicles
        """)
    files_created.append(str(ai_doc))
    
    # LangChain documentation
    langchain_doc = docs_dir / "langchain_guide.txt"
    with open(langchain_doc, "w") as f:
        f.write("""
LangChain Framework Guide

LangChain is a powerful framework for building applications with Large Language Models (LLMs).

Core Components:
1. LLMs and Chat Models - Interface with language models
2. Prompts - Templates for model inputs
3. Chains - Combine LLMs with other components
4. Agents - Use LLMs to decide actions
5. Memory - Persist state between calls
6. Document Loaders - Load data from various sources
7. Vector Stores - Store and retrieve embeddings

Key Features:
- Modular design for easy customization
- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Built-in document processing capabilities
- Vector database integrations
- Memory management for conversations
- Tool integration for agents

Common Use Cases:
- Question-answering systems
- Chatbots and conversational AI
- Document analysis and summarization
- Code generation and analysis
- Research assistants
- Content generation

Best Practices:
- Use appropriate chunk sizes for documents
- Implement proper error handling
- Monitor token usage and costs
- Use caching where appropriate
- Test with different prompt templates
        """)
    files_created.append(str(langchain_doc))
    
    return docs_dir, files_created


def main():
    """Demonstrate advanced usage of the AI Assistant."""
    
    print("=== AI Assistant Advanced Usage Example ===\n")
    
    # Create sample documents
    print("1. Creating sample documents...")
    docs_dir, files_created = create_sample_documents()
    print(f"✓ Created {len(files_created)} sample documents in '{docs_dir}'")
    
    # Initialize the AI Assistant with advanced settings
    print("\n2. Initializing AI Assistant with advanced settings...")
    assistant = AIAssistant(
        collection_name="advanced_docs",
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        use_multi_query=True,  # Enable multi-query retrieval
        use_compression=False   # Disable compression for this example
    )
    
    print("✓ AI Assistant initialized with advanced features")
    
    # Load documents from directory
    print(f"\n3. Loading documents from directory: {docs_dir}")
    doc_ids = assistant.add_documents_from_directory(str(docs_dir))
    print(f"✓ Loaded documents from directory: {len(doc_ids)} chunks added")
    
    # Load individual files
    print("\n4. Loading individual files...")
    for file_path in files_created[:2]:  # Load first 2 files individually as example
        doc_ids = assistant.add_documents_from_files(file_path)
        print(f"✓ Loaded {Path(file_path).name}: {len(doc_ids)} chunks")
    
    # Get comprehensive stats
    print("\n5. Getting comprehensive statistics...")
    stats = assistant.get_stats()
    print(f"✓ Collection: {stats['vector_store']['collection_name']}")
    print(f"✓ Total documents: {stats['vector_store'].get('document_count', 'N/A')}")
    print(f"✓ Model: {stats['assistant_config']['model_name']}")
    print(f"✓ Chunk size: {stats['assistant_config']['chunk_size']}")
    print(f"✓ Multi-query enabled: {stats['retriever_stats']['has_multi_query']}")
    
    # Advanced question answering
    print("\n6. Advanced question answering...")
    
    complex_questions = [
        "Compare Python and machine learning frameworks. Which ones work well together?",
        "How can I use LangChain to build a question-answering system with vector databases?",
        "What are the differences between supervised and unsupervised learning, and when should I use each?",
        "Explain the relationship between AI, machine learning, and deep learning with examples."
    ]
    
    for i, question in enumerate(complex_questions, 1):
        print(f"\n❓ Complex Question {i}: {question}")
        response = assistant.ask_question(question, return_sources=True)
        print(f"🤖 Answer: {response['answer']}")
        
        if 'sources' in response:
            print(f"📚 Sources used: {len(response['sources'])}")
            for j, source in enumerate(response['sources'][:2], 1):  # Show first 2 sources
                source_name = source['metadata'].get('filename', 'Unknown')
                print(f"   {j}. {source_name}")
    
    # Document search with scores
    print("\n\n7. Advanced document search...")
    
    search_queries = [
        "deep learning neural networks",
        "Python web development frameworks",
        "LangChain vector stores embeddings"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching: '{query}'")
        results = assistant.search_documents(query, k=3, with_scores=True)
        
        for i, result in enumerate(results, 1):
            filename = result['metadata'].get('filename', 'Unknown')
            score = result['score']
            content_preview = result['content'][:150].replace('\n', ' ').strip()
            print(f"   {i}. {filename} (Score: {score:.3f})")
            print(f"      Preview: {content_preview}...")
    
    # Conversational interaction with context
    print("\n\n8. Conversational interaction with context...")
    
    conversation = [
        "I'm new to AI and want to learn machine learning. Where should I start?",
        "What programming language would you recommend for a beginner?",
        "Can you suggest some specific libraries I should learn?",
        "How does LangChain fit into this learning path?"
    ]
    
    for message in conversation:
        print(f"\n💬 You: {message}")
        response = assistant.chat(message, return_sources=False)
        print(f"🤖 Assistant: {response['response']}")
    
    # Update configuration dynamically
    print("\n\n9. Dynamic configuration updates...")
    print("Current configuration:")
    current_stats = assistant.get_stats()
    print(f"   Temperature: {current_stats['assistant_config']['temperature']}")
    print(f"   Top K: {current_stats['retriever_stats']['top_k']}")
    
    # Update settings
    assistant.update_config(temperature=0.5, top_k=3)
    print("\n✓ Updated temperature to 0.5 and top_k to 3")
    
    # Test with updated settings
    test_question = "Give me a creative summary of what I can build with these technologies."
    print(f"\n❓ Test question with new settings: {test_question}")
    response = assistant.ask_question(test_question, return_sources=False)
    print(f"🤖 Answer: {response['answer']}")
    
    # Cleanup
    print("\n\n10. Cleanup...")
    assistant.clear_memory()
    print("✓ Conversational memory cleared")
    
    # Optionally clean up sample files
    cleanup = input("\nDo you want to delete the sample documents? (y/n): ").lower().strip()
    if cleanup == 'y':
        import shutil
        shutil.rmtree(docs_dir)
        print(f"✓ Deleted sample documents directory: {docs_dir}")
    else:
        print(f"✓ Sample documents preserved in: {docs_dir}")
    
    print("\n=== Advanced example completed successfully! ===")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Example interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running advanced example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)