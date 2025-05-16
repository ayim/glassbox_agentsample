from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI()
llm.invoke("Hello, world!") 