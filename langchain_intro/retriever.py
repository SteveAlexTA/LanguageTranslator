import os
import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from embedding import FastEmbedWrapper
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

dotenv.load_dotenv()

def get_qdrant_retriever(k=3):
    collection_name = "ielts_writing_task_2_evaluation"

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )

    embeddings = FastEmbedWrapper(model_name="BAAI/bge-small-en-v1.5")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})