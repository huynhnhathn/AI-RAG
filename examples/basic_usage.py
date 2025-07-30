"""
Basic Usage Example for AI Assistant
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_assistant import AIAssistant


def main():
    """Demonstrate basic usage of the AI Assistant."""
    
    print("=== AI Assistant Basic Usage Example ===\n")
    
    # Initialize the AI Assistant
    print("1. Initializing AI Assistant...")
    assistant = AIAssistant(
        collection_name="example_docs",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        chunk_size=500,
        chunk_overlap=50,
        top_k=3
    )
    
    print("✓ AI Assistant initialized successfully\n")
    
    # Add some sample text content
    print("2. Adding sample content to knowledge base...")
    
    sample_texts = [
        {
            "text": """
            Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a 
            scripting or glue language to connect existing components together.
            """,
            "metadata": {"source": "python_intro", "category": "programming"}
        },
        {
            "text": """
            Machine Learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being explicitly 
            programmed. Machine learning focuses on the development of computer programs that can 
            access data and use it to learn for themselves.
            """,
            "metadata": {"source": "ml_intro", "category": "ai"}
        },
        {
            "text": """
            LangChain is a framework for developing applications powered by language models. 
            It enables applications that are data-aware and agentic, allowing language models 
            to connect with other sources of data and interact with their environment.
            """,
            "metadata": {"source": "langchain_intro", "category": "ai_tools"}
        }
    ]
    
    for i, item in enumerate(sample_texts, 1):
        doc_ids = assistant.add_text_content(
            text=item["text"],
            metadata=item["metadata"]
        )
        print(f"✓ Added sample text {i}: {len(doc_ids)} chunks")
    
    print("\n3. Getting assistant statistics...")
    stats = assistant.get_stats()
    print(f"✓ Collection: {stats['vector_store']['collection_name']}")
    print(f"✓ Documents: {stats['vector_store'].get('document_count', 'N/A')}")
    print(f"✓ Model: {stats['assistant_config']['model_name']}")
    
    # Ask some questions
    print("\n4. Asking questions...")
    
    questions = [
        "What is Python?",
        "Tell me about machine learning",
        "What is LangChain used for?",
        "How are Python and machine learning related?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = assistant.ask_question(question, return_sources=True)
        print(f"🤖 Answer: {response['answer']}")
        
        if 'sources' in response:
            print(f"📚 Sources: {len(response['sources'])} documents used")
    
    # Demonstrate conversational chat
    print("\n\n5. Demonstrating conversational chat...")
    
    chat_messages = [
        "What programming languages are good for AI?",
        "Can you tell me more about Python's advantages?",
        "What about its disadvantages?"
    ]
    
    for message in chat_messages:
        print(f"\n💬 You: {message}")
        response = assistant.chat(message, return_sources=False)
        print(f"🤖 Assistant: {response['response']}")
    
    # Search documents
    print("\n\n6. Searching documents...")
    
    search_query = "artificial intelligence"
    print(f"\n🔍 Searching for: '{search_query}'")
    
    search_results = assistant.search_documents(
        query=search_query,
        k=2,
        with_scores=True
    )
    
    for i, result in enumerate(search_results, 1):
        print(f"\n📄 Result {i} (Score: {result['score']:.3f}):")
        print(f"Content: {result['content'][:100]}...")
        print(f"Source: {result['metadata'].get('source', 'Unknown')}")
    
    # Clear memory and show final stats
    print("\n\n7. Cleaning up...")
    assistant.clear_memory()
    print("✓ Conversational memory cleared")
    
    final_stats = assistant.get_stats()
    print(f"✓ Final document count: {final_stats['vector_store'].get('document_count', 'N/A')}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Error running example: {e}")
        sys.exit(1)