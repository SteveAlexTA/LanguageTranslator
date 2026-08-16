import os
import dotenv
from retriever import get_qdrant_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load .env
dotenv.load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY was not found in your .env file!")

retriever = get_qdrant_retriever(k=3)

review_template_str = """Your job is to use English data to answer questions related to English in language learning context.
Use the following context to answer questions. Be as detailed, informative and accurate as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

Context:
{context}"""

review_prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(review_template_str),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "Give me an explanation, a synonym and an example sentence for the word 'equilibrium'."
    print(f"Question: '{question}'\n")
    response = rag_chain.invoke(question)
    print(response)