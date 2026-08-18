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

system_prompt = """You are an official and trained IELTS Examiner.
Use the retrieved benchmark essays and examiner evaluation criteria below as your reference standards.

Context / Benchmark Essays:
{context}

Evaluate the user's IELTS essays submission:
1. Provide an estimated Overall Band Score (0.0 - 9.0) along with reasons of why the essay is graded for that score.
2. Give detailed and accurate sub-scores along with feedback across Task Achievement (TA), Coherence and Cohesion (CC), Lexical Resource (LR), and Grammatical Range and Accuracy (GRA).
3. Offer specific feedback and additional comments regarding the essay such as strengths and areas for improvement, along with suggestions for improvement based on the benchmark essays."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
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

if __name__ == "__main__":
    test_submission = """
    Prompt: Some people believe that unpaid community service should be a compulsory part of high school education. Others think students should be free to choose how they spend their free time. Discuss both views and give your opinion.

    Essay:
    Nowadays, there is a growing debate about whether teenagers should be required to complete unpaid community service as part of their schooling. While some argue this builds character and civic responsibility, others believe it infringes on students' personal freedom. This essay will discuss both perspectives before presenting my own view.

    On one hand, supporters of compulsory service argue that it teaches young people the value of contributing to society. For example, working in a shelter or tutoring younger students can build empathy and practical skills that classroom learning alone cannot provide. Countries such as Germany already require a form of civic engagement, and studies suggest participants often report a stronger sense of purpose afterward.

    On the other hand, critics contend that mandating volunteer work removes the very quality that makes it meaningful, since true volunteering should come from personal choice rather than obligation. Furthermore, many students already face heavy academic workloads, and forcing additional commitments could increase stress without necessarily building genuine civic values. Students who are passionate about music, sport, or part time work might benefit more from pursuing those interests freely.

    In my opinion, while community service has clear benefits, it should be strongly encouraged rather than strictly compulsory. Schools could offer structured opportunities and incentives, allowing students to discover the value of service without removing their autonomy entirely.

    In conclusion, both mandatory and voluntary approaches to community service have merit, but a balanced system that encourages participation without forcing it seems most likely to produce genuinely engaged citizens.
    """
    print(rag_chain.invoke(test_submission))