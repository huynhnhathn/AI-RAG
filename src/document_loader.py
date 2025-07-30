"""
Document Loading and Processing Module
"""
import os
from typing import List, Optional, Union
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredMarkdownLoader
)


class DocumentProcessor:
    """Handles document loading, processing, and chunking for the AI assistant."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_token_splitter: bool = False):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_token_splitter: Whether to use token-based splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        if use_token_splitter:
            self.text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """
        Load a single text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            return self._add_metadata(documents, file_path)
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_pdf_file(self, file_path: str) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self._add_metadata(documents, file_path)
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def load_csv_file(self, file_path: str, csv_args: dict = None) -> List[Document]:
        """
        Load a CSV file.
        
        Args:
            file_path: Path to the CSV file
            csv_args: Additional arguments for CSV loader
            
        Returns:
            List of Document objects
        """
        try:
            csv_args = csv_args or {}
            loader = CSVLoader(file_path, **csv_args)
            documents = loader.load()
            return self._add_metadata(documents, file_path)
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {e}")
            return []
    
    def load_json_file(self, file_path: str, jq_schema: str = None) -> List[Document]:
        """
        Load a JSON file.
        
        Args:
            file_path: Path to the JSON file
            jq_schema: JQ schema for extracting data
            
        Returns:
            List of Document objects
        """
        try:
            if jq_schema:
                loader = JSONLoader(file_path, jq_schema=jq_schema)
            else:
                loader = JSONLoader(file_path, jq_schema='.')
            documents = loader.load()
            return self._add_metadata(documents, file_path)
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def load_markdown_file(self, file_path: str) -> List[Document]:
        """
        Load a Markdown file.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            List of Document objects
        """
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            return self._add_metadata(documents, file_path)
        except Exception as e:
            print(f"Error loading Markdown file {file_path}: {e}")
            return []
    
    def load_directory(self, 
                      directory_path: str, 
                      glob_pattern: str = "**/*",
                      exclude_patterns: List[str] = None) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of Document objects
        """
        exclude_patterns = exclude_patterns or ["*.pyc", "*.git*", "__pycache__/*"]
        
        all_documents = []
        
        # Define loaders for different file types
        loaders = {
            "*.txt": TextLoader,
            "*.md": UnstructuredMarkdownLoader,
            "*.pdf": PyPDFLoader,
            "*.csv": CSVLoader,
            "*.json": JSONLoader,
        }
        
        for pattern, loader_class in loaders.items():
            try:
                loader = DirectoryLoader(
                    directory_path,
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=True
                )
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading files with pattern {pattern}: {e}")
        
        return all_documents
    
    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Create documents from raw text.
        
        Args:
            text: Raw text content
            metadata: Optional metadata for the document
            
        Returns:
            List of Document objects
        """
        metadata = metadata or {}
        document = Document(page_content=text, metadata=metadata)
        return [document]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        try:
            chunked_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk information to metadata
            for i, doc in enumerate(chunked_docs):
                doc.metadata['chunk_id'] = i
                doc.metadata['chunk_size'] = len(doc.page_content)
            
            return chunked_docs
        except Exception as e:
            print(f"Error chunking documents: {e}")
            return documents
    
    def process_documents(self, 
                         file_paths: Union[str, List[str]] = None,
                         directory_path: str = None,
                         text_content: str = None,
                         chunk: bool = True) -> List[Document]:
        """
        Process documents from various sources.
        
        Args:
            file_paths: Single file path or list of file paths
            directory_path: Directory to process
            text_content: Raw text content
            chunk: Whether to chunk the documents
            
        Returns:
            List of processed documents
        """
        all_documents = []
        
        # Process individual files
        if file_paths:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            
            for file_path in file_paths:
                documents = self._load_single_file(file_path)
                all_documents.extend(documents)
        
        # Process directory
        if directory_path:
            documents = self.load_directory(directory_path)
            all_documents.extend(documents)
        
        # Process text content
        if text_content:
            documents = self.load_from_text(text_content)
            all_documents.extend(documents)
        
        # Chunk documents if requested
        if chunk and all_documents:
            all_documents = self.chunk_documents(all_documents)
        
        return all_documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file based on its extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return self.load_text_file(str(file_path))
        elif extension == '.pdf':
            return self.load_pdf_file(str(file_path))
        elif extension == '.csv':
            return self.load_csv_file(str(file_path))
        elif extension == '.json':
            return self.load_json_file(str(file_path))
        elif extension in ['.md', '.markdown']:
            return self.load_markdown_file(str(file_path))
        else:
            # Try to load as text file
            return self.load_text_file(str(file_path))
    
    def _add_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """Add file metadata to documents."""
        file_path = Path(file_path)
        
        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'filename': file_path.name,
                'file_type': file_path.suffix,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })
        
        return documents