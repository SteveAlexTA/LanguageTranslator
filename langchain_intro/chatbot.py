import os
import dotenv
from retriever import get_qdrant_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
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

system_prompt = """You are an official, certified and trained IELTS Examiner.
You are provided with reference benchmark essays, scoring criteria, and examiner rubrics below.
Use this context STRICTLY as your internal standard and calibration baseline for accurate band scoring.
Internal Scoring Rubrics & Benchmark Standards:
{context}
---
CRITICAL EXAMINER INSTRUCTIONS:
- NEVER mention or reference "benchmark essays", "retrieved essays", "reference documents", or "context" in your output.
- Address the user directly as a candidate receiving an official IELTS Writing report.
- Be objective, constructive, and concise. Avoid unnecessary meta-commentary.
Provide your evaluation in the following structured format:
1. Overall Band Score: State the score (0.0 - 9.0) with a 2-3 sentence executive summary.
2. Criteria Breakdown:
   - Task Achievement (TA): [Band Score] - Specific assessment of prompt response, position clarity, and argument development.
   - Coherence and Cohesion (CC): [Band Score] - Specific assessment of paragraphing, flow, and linking devices.
   - Lexical Resource (LR): [Band Score] - Specific assessment of vocabulary range, collocations, and precision.
   - Grammatical Range and Accuracy (GRA): [Band Score] - Specific assessment of sentence diversity, syntax, and accuracy.
3. Key Strengths: 2 bullet points highlighting standout elements.
4. Actionable Suggestions for Improvement: 2-3 concrete tips with specific sentence rewrites to push the essay to the next band level."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    google_api_key=gemini_key,
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)

