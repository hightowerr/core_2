"""Question-answering engine module."""

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class QAEngine:
    """Handles question-answering using OpenAI models with advanced retrieval."""
    
    def __init__(self, config):
        """
        Initialize QA Engine with GPT-4o.
        
        Args:
            config (Config): Configuration object
        """
        # Use GPT-4o explicitly with API key from config
        self.model = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.1,  # More focused responses
            max_tokens=2000,   # Limit response length
            openai_api_key=config.get_openai_api_key()
        )
    
    def create_retrieval_chain(self, vectorstore):
        """
        Create a retrieval QA chain with a custom prompt.
        
        Args:
            vectorstore (Chroma): Vector store containing processed content.
        
        Returns:
            RetrievalQA: Configured retrieval QA chain
        """
        # Custom prompt template for more contextual responses
        prompt_template = """You are analyzing content from the URL specified in the question.

        Use the following pieces of context to answer the question. Make sure your answer is specifically about the content from the URL, not general knowledge. If the answer cannot be found in the URL's content, say so clearly.

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 3 most relevant documents
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return retrieval_qa
    
    def ask_question(self, vectorstore, question, url):
        """
        Answer a question based on the processed content using retrieval chain.
        
        Args:
            vectorstore (Chroma): Vector store containing processed content.
            question (str): Question to ask about the content.
            url (str): The source URL being analyzed.
        
        Returns:
            dict: Response with answer and source documents
        """
        if not vectorstore or not question:
            return {
                "answer": "Insufficient information to answer the question.",
                "source_documents": []
            }
        
        try:
            # Create retrieval chain
            retrieval_chain = self.create_retrieval_chain(vectorstore)
            
            # Include URL context in the question
            contextualized_question = f"For the content from URL: {url}\n\n{question}"
            
            # Run the query
            result = retrieval_chain({"query": contextualized_question})
            
            return {
                "answer": result['result'],
                "source_documents": result.get('source_documents', [])
            }
        
        except Exception as e:
            print(f"Error in question answering: {e}")
            return {
                "answer": "Could not generate an answer.",
                "source_documents": []
            }
