from typing import List
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from langchain_core.documents import Document

class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(next(self.model.embed([text])))