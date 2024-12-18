"""Configuration management for the web Q&A application."""

import os
from dotenv import load_dotenv

class Config:
    """Application configuration class."""
    
    def __init__(self, api_key=None):
        """Initialize configuration, loading environment variables."""
        load_dotenv()
        
        # OpenAI configuration
        self.openai_api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        # Vector store configuration
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 2000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def get_openai_model(self):
        """Get the configured OpenAI model."""
        return self.openai_model

    def get_text_splitter_config(self):
        """Get text splitter configuration."""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
    def get_openai_api_key(self):
        """Get the OpenAI API key."""
        return self.openai_api_key
