# rag_web.py
# Local web interface for the RAG app — runs in your browser at localhost:8501
# Nothing is deployed or made public — all data stays on your machine

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Course Notes Q&A", page_icon="📚")
st.title("📚 AI Course Notes Q&A")
st.caption("Ask questions about your course notes — answers are grounded only in your documents")

# ── CACHED SETUP ─────────────────────────────────────────────────────────────
# @st.cache_resource tells Streamlit to run this function only ONCE and reuse
# the result on every subsequent interaction. Without this, Streamlit would
# reload the vector store and rebuild the chain on every single message sent,
# making it extremely slow.
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load from disk if vector store already exists, otherwise build it
    if os.path.exists("chroma_db"):
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )
    else:
        loader = PyPDFDirectoryLoader("docs/")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context from the course notes.
If the answer isn't directly and clearly in the context, respond with only:
"I don't see that in your notes." — do not elaborate further.

Context:
{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both chain and retriever so we can display sources
    return chain, retriever

# Load everything once on startup
chain, retriever = load_chain()

# ── CHAT HISTORY ─────────────────────────────────────────────────────────────
# st.session_state persists data across interactions within the same session.
# Without this, the chat history would reset every time the user sends a message.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── CHAT INPUT ───────────────────────────────────────────────────────────────
# st.chat_input renders the text box at the bottom of the page.
# The := operator assigns the input value AND checks if it's not empty
if question := st.chat_input("Ask a question about your course notes..."):

    # Display the user's question in the chat
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Run the RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):

            # Get answer from chain
            answer = chain.invoke(question)

            # Get source documents for citation
            source_docs = retriever.invoke(question)

            # Format source citations
            sources = "\n".join([
                f"- {doc.metadata.get('source', 'unknown').split('/')[-1]} (page {doc.metadata.get('page', '?')})"
                for doc in source_docs
            ])

            # Build full response with sources appended
            full_response = f"{answer}\n\n---\n**Sources:**\n{sources}"

            st.markdown(full_response)

    # Save assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })