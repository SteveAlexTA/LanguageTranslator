import os
import dotenv
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

dotenv.load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not gemini_key or not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing GEMINI_API_KEY, QDRANT_URL, or QDRANT_API_KEY in .env file!")

json_path = "data/vocabulary_test.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found at '{json_path}'")

# Load .json data
with open(json_path, "r", encoding="utf-8") as f:
    vocab_items = json.load(f)  

# Format into LangChain documents
docs = []
for item in vocab_items:
    content = (
        f"Word: {item['word']}\n"
        f"Part of Speech: {item['pos']}\n"
        f"Difficulty: {item['difficulty']}\n"
        f"Theme: {item['theme']}\n"
        f"Synonyms: {item['synonyms']}\n"
        f"Definition: {item['definition_en']}\n"
        f"Example Sentence: {item['example_sentence']}"
    )
    docs.append(Document(page_content=content, metadata={"word": item["word"], "theme": item["theme"]}))

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

vector_store = QdrantVectorStore(
    client=client,
    collection_name="toefl_vocabulary",
    embedding=embeddings,
)

vector_store.add_documents(docs)