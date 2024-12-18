import os
from langchain_openai import ChatOpenAI
from utils import fetch_web_content, process_content, ask_question

def main():
    # Initialize OpenAI model
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Get URL from user
    url = input("Enter the URL of the web page to load: ")
    
    # Fetch web content from the provided URL
    content = fetch_web_content(url)
    
    if content:
        # Process the content into a vector store
        vectorstore = process_content(content)
        
        if vectorstore:
            # Allow user to ask questions about the content
            while True:
                question = input("Ask a question about the web page (or 'quit' to exit): ")
                
                if question.lower() == 'quit':
                    break
                
                # Get answer from the model
                answer = ask_question(model, vectorstore, question)
                print("\nAnswer:", answer)
        else:
            print("Failed to process web page content.")
    else:
        print("Failed to fetch web page content.")

if __name__ == "__main__":
    main()
