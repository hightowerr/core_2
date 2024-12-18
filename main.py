import os
from langchain_openai import ChatOpenAI
from utils import fetch_web_content, process_content, ask_question

def main():
    # Initialize OpenAI model
    url = input("Enter the URL of the web page to load: ")
    
    # Fetch web content from the provided URL
    content = fetch_web_content(url)
    
    if content:
        print("Web Page Content:")
        print(content)
    else:
        print("Failed to fetch web page content.")

if __name__ == "__main__":
    main()
