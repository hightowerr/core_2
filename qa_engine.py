"""Question-answering engine module."""

from langchain_openai import ChatOpenAI

class QAEngine:
    """Handles question-answering using OpenAI models."""
    
    def __init__(self, config):
        """
        Initialize QA Engine.
        
        Args:
            config (Config): Configuration object
        """
        self.model = ChatOpenAI(model=config.get_openai_model())
    
    def ask_question(self, vectorstore, question):
        """
        Answer a question based on the processed content.
        
        Args:
            vectorstore (Chroma): Vector store containing processed content.
            question (str): Question to ask about the content.
        
        Returns:
            str: Model's answer to the question.
        """
        if not vectorstore or not question:
            return "Insufficient information to answer the question."
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever()
        try:
            relevant_docs = retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return "Could not retrieve relevant information."
        
        # Prepare context
        context = " ".join([doc.page_content for doc in relevant_docs])
        
        # Construct prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate answer
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Could not generate an answer."
