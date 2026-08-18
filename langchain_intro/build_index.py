import os
import time
import dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams 

dotenv.load_dotenv()

# Load URL and API keys from .env
gemini_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not gemini_key or not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing GEMINI_API_KEY, QDRANT_URL, or QDRANT_API_KEY in .env file")

# Define path to CSV file
csv_path = "data/train.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at '{csv_path}'")

collection_name = "ielts_writing_task_2_evaluation"

# Connect to Qdrant Client 
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=120,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=gemini_key
)

# Create collection if no collection exists
if not client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' not found. Creating collection...")
    sample_vector = embeddings.embed_query("test dimension verification")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created with vector size {len(sample_vector)}.")
else:
    print(f"Found existing collection '{collection_name}'. Proceeding to upload.")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ielts_writing_task_2_evaluation",
    embedding=embeddings,
)

# Process CSV row-by-row in streaming batches 
batch_size = 40
chunk_count = 0
for df_chunk in pd.read_csv(csv_path, chunksize=batch_size):
    chunk_count += 1
    print(f"Processing chunk {chunk_count} with {len(df_chunk)} rows...")

    docs = []
    for _, row in df_chunk.iterrows():
        prompt_text = str(row.get("prompt", "")).strip()
        essay_text = str(row.get("essay", "")).strip()
        evaluation_text = str(row.get("evaluation", "")).strip()
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
            "source": "data/train.csv",
            "band": band_score,
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    # Upload batch to Qdrant 
    if docs:
        vector_store.add_documents(docs)
        print(f"Uploaded {len(docs)} documents to Qdrant collection 'ielts_writing_task_2_evaluation'.")
        time.sleep(1)
