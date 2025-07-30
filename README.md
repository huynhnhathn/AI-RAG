# AI Assistant with LangChain, OpenAI, and Qdrant

A powerful AI assistant built with LangChain that retrieves context from a Qdrant vector database and uses OpenAI's language models for intelligent question answering and conversational AI.

## Features

- 🤖 **Intelligent Q&A**: Ask questions and get contextual answers from your documents
- 💬 **Conversational AI**: Maintain context across conversations with memory
- 📚 **Document Processing**: Load and process various document formats (TXT, PDF, MD, CSV, JSON)
- 🔍 **Advanced Retrieval**: Multi-query and contextual compression retrieval options
- 📊 **Vector Search**: Semantic search with relevance scores
- ⚙️ **Configurable**: Customizable chunk sizes, models, and retrieval parameters
- 🚀 **Easy Setup**: Simple installation and configuration

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   AI Assistant   │    │   OpenAI LLM    │
│                 │    │                  │    │                 │
│ • PDF Files     │───▶│ • Document       │───▶│ • GPT-3.5/4     │
│ • Text Files    │    │   Processing     │    │ • Embeddings    │
│ • Markdown      │    │ • Vector Storage │    │ • Chat Models   │
│ • CSV/JSON      │    │ • Retrieval      │    │                 │
└─────────────────┘    │ • Q&A Chains     │    └─────────────────┘
                       └──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Qdrant Vector   │
                       │ Database        │
                       │                 │
                       │ • Embeddings    │
                       │ • Similarity    │
                       │ • Metadata      │
                       └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Qdrant instance (local or cloud)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your configuration:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_api_key_if_needed
   QDRANT_COLLECTION_NAME=ai_assistant_docs
   ```

4. **Start Qdrant** (if running locally):
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or using Docker Compose
   docker-compose up -d
   ```

## Quick Start

### Basic Usage

```python
from src.ai_assistant import AIAssistant

# Initialize the assistant
assistant = AIAssistant(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    chunk_size=1000,
    top_k=4
)

# Add some text content
assistant.add_text_content(
    text="Python is a versatile programming language...",
    metadata={"source": "python_guide", "category": "programming"}
)

# Ask questions
response = assistant.ask_question("What is Python?")
print(response["answer"])

# Have a conversation
chat_response = assistant.chat("Tell me more about Python's advantages")
print(chat_response["response"])
```

### Loading Documents

```python
# Load from files
assistant.add_documents_from_files([
    "docs/guide.pdf",
    "docs/manual.txt"
])

# Load from directory
assistant.add_documents_from_directory("./documents")

# Search documents
results = assistant.search_documents(
    query="machine learning",
    k=5,
    with_scores=True
)
```

## Examples

### Run Basic Example

```bash
python examples/basic_usage.py
```

This example demonstrates:
- Initializing the AI assistant
- Adding text content to the knowledge base
- Asking questions and getting contextual answers
- Conversational chat with memory
- Document search with relevance scores

### Run Advanced Example

```bash
python examples/advanced_usage.py
```

This example shows:
- Loading documents from files and directories
- Advanced retrieval configurations
- Complex question answering
- Dynamic configuration updates
- Comprehensive statistics and monitoring

## Configuration Options

### AI Assistant Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | str | `"ai_assistant_docs"` | Qdrant collection name |
| `model_name` | str | `"gpt-3.5-turbo"` | OpenAI model to use |
| `temperature` | float | `0.0` | Model creativity (0.0-1.0) |
| `chunk_size` | int | `1000` | Document chunk size |
| `chunk_overlap` | int | `200` | Overlap between chunks |
| `top_k` | int | `4` | Number of documents to retrieve |
| `use_multi_query` | bool | `False` | Enable multi-query retrieval |
| `use_compression` | bool | `False` | Enable contextual compression |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `QDRANT_URL` | No | Qdrant instance URL (default: localhost:6333) |
| `QDRANT_API_KEY` | No | Qdrant API key (if required) |
| `QDRANT_COLLECTION_NAME` | No | Collection name (default: ai_assistant_docs) |

## API Reference

### AIAssistant Class

#### Methods

**`add_documents_from_files(file_paths, chunk=True)`**
- Load documents from file paths
- Supports: TXT, PDF, MD, CSV, JSON
- Returns: List of document IDs

**`add_documents_from_directory(directory_path, chunk=True)`**
- Load all supported files from a directory
- Recursively processes subdirectories
- Returns: List of document IDs

**`add_text_content(text, metadata=None, chunk=True)`**
- Add raw text content directly
- Optional metadata for categorization
- Returns: List of document IDs

**`ask_question(question, return_sources=True)`**
- Ask a question and get contextual answer
- Returns: Dictionary with answer and sources

**`chat(message, return_sources=True)`**
- Conversational chat with memory
- Maintains context across messages
- Returns: Dictionary with response and sources

**`search_documents(query, k=None, with_scores=False)`**
- Search for relevant documents
- Optional relevance scores
- Returns: List of documents with metadata

**`get_stats()`**
- Get comprehensive statistics
- Returns: Configuration and usage information

**`clear_memory()`**
- Clear conversational memory
- Resets chat context

**`update_config(**kwargs)`**
- Update configuration dynamically
- Supports: temperature, top_k, chunk_size, etc.

## Supported Document Formats

- **Text Files** (`.txt`): Plain text documents
- **PDF Files** (`.pdf`): Portable Document Format
- **Markdown** (`.md`, `.markdown`): Markdown formatted text
- **CSV Files** (`.csv`): Comma-separated values
- **JSON Files** (`.json`): JavaScript Object Notation

## Advanced Features

### Multi-Query Retrieval

Improves retrieval accuracy by generating multiple query variations:

```python
assistant = AIAssistant(use_multi_query=True)
```

### Contextual Compression

Filters retrieved content for relevance:

```python
assistant = AIAssistant(use_compression=True)
```

### Custom Prompts

Customize the AI's behavior with custom prompts:

```python
custom_prompt = """
You are a helpful assistant specialized in technical documentation.
Always provide code examples when relevant.

Context: {context}
Question: {question}
Answer:
"""

qa_chain = assistant.retriever.create_qa_chain(custom_prompt=custom_prompt)
```

## Monitoring and Debugging

### Get Statistics

```python
stats = assistant.get_stats()
print(f"Documents: {stats['vector_store']['document_count']}")
print(f"Model: {stats['assistant_config']['model_name']}")
print(f"Collection: {stats['vector_store']['collection_name']}")
```

### Search with Scores

```python
results = assistant.search_documents(
    query="your query",
    with_scores=True
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set in environment or `.env` file
   - Verify the API key is valid and has sufficient credits

2. **Qdrant Connection Error**
   - Check if Qdrant is running on the specified URL
   - Verify network connectivity and firewall settings

3. **Document Loading Issues**
   - Ensure file paths are correct and files exist
   - Check file permissions and encoding (UTF-8 recommended)

4. **Memory Issues with Large Documents**
   - Reduce `chunk_size` parameter
   - Process documents in smaller batches

### Performance Optimization

- **Chunk Size**: Smaller chunks (500-800) for better precision, larger chunks (1000-1500) for more context
- **Top K**: Increase for more comprehensive answers, decrease for faster responses
- **Temperature**: Lower values (0.0-0.3) for factual responses, higher (0.5-0.8) for creative responses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the examples directory for usage patterns
- Review the API documentation above

## Changelog

### v1.0.0
- Initial release
- Basic Q&A functionality
- Document loading and processing
- Vector storage with Qdrant
- Conversational AI with memory
- Multi-format document support