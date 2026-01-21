import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import ollama


st.set_page_config(page_title="Jarvis", layout="centered")
st.title("Jarvis (Streamlit + Pinecone + phi3)")

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "jarvis-index")

if not PINECONE_API_KEY:
    st.error("Pinecone API key not found. Please check")
    st.stop()


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def upload_docs():
    docs = [
        "Company Leave Policy: Employees get 20 paid leaves per year. Sick leave is 10 days.",
        "Onboarding Steps: Step1 HR docs, Step2 Laptop setup, Step3 Laptop + Email setup, Step4 Team introduction, Step5 Project assignment.",
        "Office Timings: Monday to Friday, 9:00 AM to 6:00 PM. Hybrid model allowed.",
        "Jarvis System: This assistant uses Pinecone vector search to retrieve relevant context and a self-hosted LLM to answer."
    ]

    vectors = []
    for i, text in enumerate(docs):
        emb = embed_model.encode(text).tolist()
        vectors.append((f"doc-{i}", emb, {"text": text}))

    index.upsert(vectors=vectors)

if "uploaded" not in st.session_state:
    upload_docs()
    st.session_state.uploaded = True

def retrieve_context(query, top_k=3):
    q_emb = embed_model.encode(query).tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    context = "\n".join([m["metadata"]["text"] for m in res["matches"]])
    return context


if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

query = st.chat_input("Ask Jarvis something...")

if query:
    st.session_state.chat.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    context = retrieve_context(query)

    prompt = f"""
You are Jarvis, a helpful assistant.
Answer ONLY using the context.

Context:
{context}

Question:
{query}

Answer:
"""

    
    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)
