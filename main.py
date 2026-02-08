# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import os

# -------------------------------
# 1. Initialize FastAPI
# -------------------------------
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all devices
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# -------------------------------
# 2. Request model
# -------------------------------
class Msg(BaseModel):
    message: str

# -------------------------------
# 3. Ensure knowledge base exists
# -------------------------------
if not os.path.exists("train.txt"):
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write(
            "Hello! This is a sample knowledge base.\n"
            "FastAPI is used to build APIs.\n"
            "LangChain helps process and split text.\n"
            "This text can be used to answer questions with RAG.\n"
        )
    print("Created sample train.txt file.")

# -------------------------------
# 4. Load and split documents
# -------------------------------
loader = TextLoader("train.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# -------------------------------
# 5. Create embeddings + vectorstore
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -------------------------------
# 6. Hugging Face GPT-2 API helper
# -------------------------------

HF_API_KEY = os.getenv("HF_API_KEY")


HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"

def query_hf(prompt: str):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    data = response.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    return "Sorry, could not generate a response."

# -------------------------------
# 7. Hybrid RAG function
# -------------------------------
def hybrid_rag(question: str):
    # Get relevant docs from vectorstore
    results = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in results]) if results else ""

    prompt = f"""
You are a helpful AI assistant.

Below is some context that may or may not be useful:

{context}

Question: {question}

Rules:
- Use context if it answers the question
- Otherwise use your own knowledge
- Never say "I don't know"

Final Answer:
"""
    return query_hf(prompt)

# -------------------------------
# 8. API endpoint
# -------------------------------
@app.post("/chat")
def chat(msg: Msg):
    reply = hybrid_rag(msg.message)
    return {"reply": reply}
