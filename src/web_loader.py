"""Web content loading and processing module."""

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class WebContentLoader:
    """Handles web content fetching and processing."""
    
    def __init__(self, config):
        """
        Initialize WebContentLoader.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.get_openai_api_key())
    
    def fetch_content(self, url):
        """
        Fetch web page content from the given URL.
        
        Args:
            url (str): The URL of the web page to fetch.
        
        Returns:
            str: The text content of the web page, or None if fetch fails.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching web content: {e}")
            return None
    
    def process_content(self, content):
        """
        Process web page content into a vector store.
        
        Args:
            content (str): Raw web page content.
        
        Returns:
            Chroma: Vector store of processed content, or None if processing fails.
        """
        if not content:
            return None
        
        # Clean HTML content
        soup = BeautifulSoup(content, 'html.parser')
        clean_text = soup.get_text()
        
        # Split text into chunks
        splitter_config = self.config.get_text_splitter_config()
        text_splitter = RecursiveCharacterTextSplitter(**splitter_config)
        texts = text_splitter.split_text(clean_text)
        
        # Create vector store
        try:
            vectorstore = Chroma.from_texts(
                texts, 
                embedding=self.embeddings
            )
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
