import os
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
dotenv.load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY was not found in your .env file!")

# Initialize ChatGoogleGenerativeAI
chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=gemini_key
)

messages = [
    SystemMessage(
        content="You're an expert assistant knowledgeable about Academic IELTS and TOEFL exam preparation."
    ),
    HumanMessage(
        content="What are the main differences between IELTS Academic Writing Task 1 and TOEFL Writing Task 1?"
    )
]

print("Sending request to Google Gemini API...\n")
response = chat_model.invoke(messages)

print("Response from Google Gemini API:\n")
print(response.content)