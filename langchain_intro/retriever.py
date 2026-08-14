import os
import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

dotenv.load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY missing from .env file!")

def get_qdrant_retriever(csv_path="data/toefl_essential_vocabulary.csv", collection_name="toefl_vocabulary", k=3):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=gemini_key
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})