import os
import time
import dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from embedding import FastEmbedWrapper
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams 

dotenv.load_dotenv()

# Load URL and API keys from .env
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env file")

# Define path to CSV file
csv_path = "data/test.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at '{csv_path}'")

collection_name = "ielts_writing_task_2_evaluation"

# Initialize embeddings and verify vector size
embeddings = FastEmbedWrapper(model_name="BAAI/bge-small-en-v1.5")
sample_vector = embeddings.embed_query("test dimension verification")
vector_size = len(sample_vector)

# Connect to Qdrant Client 
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=120,
)

# Create collection if no collection exists
if not client.collection_exists(collection_name):
    print(f"Creating collection '{collection_name}' with vector size {vector_size}.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
else:
    collection_info = client.get_collection(collection_name)
    existing_vector_size = collection_info.config.params.vectors.size
    if existing_vector_size != vector_size:
        print(f"Vector size mismatch ({existing_vector_size} vs {vector_size}), recreating collection.")
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists with matching vector size {vector_size}.")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ielts_writing_task_2_evaluation",
    embedding=embeddings,
)

# Process data row-by-row in streaming batches 
batch_size = 50
chunk_count = 0
for df_chunk in pd.read_csv(csv_path, chunksize=batch_size):
    chunk_count += 1
    docs = []
    for _, row in df_chunk.iterrows():
        prompt_text = str(row.get("prompt", "")).strip()
        essay_text = str(row.get("essay", "")).strip()
        evaluation_text = str(row.get("evaluation", "")).strip()[:1200]
        try:
            band_score = float(row.get("band", 0.0))
        except (ValueError, TypeError):
            band_score = 0.0

        # Structure text for LLM
        page_content = (
            f"IELTS Prompt: {prompt_text}\n\n"
            f"Candidate Essay: {essay_text}\n\n"
            f"Examiner Evaluation & Criteria Breakdown: {evaluation_text}\n\n"
            f"Band Score: {band_score}"
        )

        # Store fillaterable attributes in metadata
        metadata = {
            "source": csv_path,
            "band": band_score,
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    if docs:
        vector_store.add_documents(docs)
        print(f"Uploaded batch {chunk_count} ({len(docs)} documents)")

