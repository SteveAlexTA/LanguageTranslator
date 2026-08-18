# LanguageTranslator 

LanguageTranslator is a small RAG assistant that answers language-learning questions by combining LangChain, Google Gemini, and Qdrant.

Features

- Retrieval-Augmented Generation (RAG)
- Domain-focused (vocabulary / language learning)
- Uses Google Gemini AI models for embeddings and chat
- Qdrant vector store for semantic search
- Small, easy-to-run example scripts

Tech stack

- Language: Python
- RAG framework: LangChain
- LLM & embeddings: Google Gemini 
- Vector DB: Qdrant
- Packaging: Docker 

Project structure

```
LanguageTranslator
│
├── Dockerfile
├── compose.yml
├── requirements.txt
├── data/                  # vocabulary JSON (data/vocabulary_test.json)
└── langchain_intro/
    ├── build_index.py     # create embeddings and push to Qdrant
    ├── chatbot.py         # example RAG chain (sample question)
    └── retriever.py       # helper to create a Qdrant retriever
```

How it works

1. Build embeddings from JSON/CSV datasets
3. Run the chatbot: it retrieves relevant docs and asks Gemini to answer using that context.


