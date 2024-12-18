import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def fetch_web_content(url):
    """
    Fetch the web page content from the given URL.
    
    Args:
        url (str): The URL of the web page to fetch.
    
    Returns:
        str: The text content of the web page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching web content: {e}")
        return None

def process_content(content):
    """
    Process the web page content and load it into a vector store.
    
    Args:
        content (str): The text content of the web page.
    
    Returns:
        Chroma: A vector store containing the processed content.
    """
    if not content:
        return None
    
    # Use BeautifulSoup to clean the HTML content
    soup = BeautifulSoup(content, 'html.parser')
    clean_text = soup.get_text()
    
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(clean_text)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts, 
        embedding=embeddings
    )
    
    return vectorstore

def ask_question(model, processed_content, question):
    """
    Use the OpenAI model to answer a question based on the processed content.
    
    Args:
        model (ChatOpenAI): The language model to use for answering.
        processed_content (Chroma): The vector store containing the processed content.
        question (str): The question to ask about the content.
    
    Returns:
        str: The model's answer to the question.
    """
    if not processed_content or not question:
        return "Insufficient information to answer the question."
    
    # Retrieve relevant documents from the vector store
    retriever = processed_content.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Prepare the context for the model
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    # Construct the prompt with context and question
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Generate the answer using the language model
    response = model.invoke(prompt)
    
    return response.content
