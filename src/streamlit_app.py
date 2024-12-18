"""Streamlit UI for Web Content Q&A Application."""

import streamlit as st
from config import Config
from web_loader import WebContentLoader
from qa_engine import QAEngine

def main():
    """
    Streamlit application for web content Q&A.
    Provides a web interface for loading web content and asking questions.
    """
    st.title("Web Content Q&A")
    
    # Initialize configuration and components
    config = Config()
    web_loader = WebContentLoader(config)
    qa_engine = QAEngine(config)
    
    # URL input
    url = st.text_input("Enter the URL of the web page to load:")
    
    # Process content when URL is provided
    if url:
        with st.spinner('Fetching and processing web content...'):
            # Fetch web content
            content = web_loader.fetch_content(url)
            
            if content:
                # Process content into vector store
                vectorstore = web_loader.process_content(content)
                
                if vectorstore:
                    # Question input
                    question = st.text_input("Ask a question about the web page:")
                    
                    # Ask question button
                    if st.button("Get Answer"):
                        if question:
                            with st.spinner('Generating answer...'):
                                # Get answer from QA engine
                                result = qa_engine.ask_question(vectorstore, question)
                                
                                # Display answer
                                st.subheader("Answer")
                                st.write(result['answer'])
                                
                                # Display source documents
                                if result['source_documents']:
                                    st.subheader("Source Documents")
                                    for i, doc in enumerate(result['source_documents'], 1):
                                        st.text_area(f"Source {i}", value=doc.page_content, height=100)
                        else:
                            st.warning("Please enter a question.")
                else:
                    st.error("Failed to process web page content.")
            else:
                st.error("Failed to fetch web page content.")

if __name__ == "__main__":
    main()
