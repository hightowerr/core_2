"""Main application entry point."""

from config import Config
from web_loader import WebContentLoader
from qa_engine import QAEngine

def main():
    """
    Main application logic for web content Q&A.
    
    Fetches web content, processes it, and allows 
    interactive question-answering.
    """
    # Initialize configuration
    config = Config()
    
    # Initialize components
    web_loader = WebContentLoader(config)
    qa_engine = QAEngine(config)
    
    # Get URL from user
    url = input("Enter the URL of the web page to load: ")
    
    # Fetch web content
    content = web_loader.fetch_content(url)
    
    if content:
        # Process content into vector store
        vectorstore = web_loader.process_content(content)
        
        if vectorstore:
            # Interactive Q&A loop
            while True:
                question = input("Ask a question about the web page (or 'quit' to exit): ")
                
                if question.lower() == 'quit':
                    break
                
                # Get and print answer
                answer = qa_engine.ask_question(vectorstore, question)
                print("\nAnswer:", answer)
        else:
            print("Failed to process web page content.")
    else:
        print("Failed to fetch web page content.")

if __name__ == "__main__":
    main()
