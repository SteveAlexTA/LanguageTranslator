# LanguageTranslator - RAG Assistant

LanguageTranslator is a small LLM assistant that answers IELTS learning questions by RAG (Retrieval-Augmented Generation) using LangChain, Google Gemini, and Qdrant.

## Features

- Retrieval-Augmented Generation (RAG)
- Domain-focused (vocabulary / language learning)
- Uses Gemini AI for embeddings and chat
- Qdrant vector store
- Streamlit UI 


## Tech Stack

- Language: Python
- RAG framework: LangChain
- LLM & embeddings: Google Gemini 
- Vector DB: Qdrant 
- UI: Streamlit
- Packaging: Docker
## Project structure

```
LanguageTranslator
│
├── Dockerfile
├── compose.yml
├── requirements.txt
├── data/                 
└── langchain_intro/
    ├── build_index.py    
    ├── chatbot.py        
    ├── retriever.py       
    └── ui.py          
```

## How it works

1. Build embeddings from your vocabulary JSON.
2. Store embeddings in Qdrant.
3. The Streamlit UI or chatbot retrieves relevant documents and asks LLM to answer using that context.


