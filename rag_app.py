# rag_app.py
import os
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

# ── STEP 1: SET UP EMBEDDING MODEL ──────────────────────────────────────────
# Always needed whether loading from disk or building fresh
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── STEP 2: LOAD OR BUILD VECTOR STORE ──────────────────────────────────────
# If chroma_db folder already exists, load it from disk (fast)
# If not, load PDFs, split into chunks, build vector store, and save to disk
if os.path.exists("chroma_db"):
    print("Loading existing vector store from disk...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
else:
    # Only runs on the very first launch
    print("Loading documents...")
    loader = PyPDFDirectoryLoader("docs/")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from your PDFs")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # each chunk is ~1000 characters
        chunk_overlap=200   # 200 character overlap between chunks to avoid losing context at boundaries
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Building vector store (this may take a minute)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"   # saves to disk so future runs skip this step
    )

print("Vector store ready")

# ── STEP 3: SET UP THE RETRIEVER ─────────────────────────────────────────────
# Searches the vector store for the 4 most relevant chunks per question
# using cosine similarity between the question vector and stored chunk vectors
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── STEP 4: SET UP CLAUDE ────────────────────────────────────────────────────
# temperature=0 means consistent, deterministic answers — good for Q&A
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0
)

# ── STEP 5: BUILD THE PROMPT TEMPLATE ───────────────────────────────────────
# {context} gets replaced with retrieved chunks
# {question} gets replaced with whatever the user typed
# telling Claude to admit when the answer isn't in the notes
# prevents hallucination — it won't invent an answer
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context from the course notes.
If the answer isn't directly and clearly in the context, respond with only:
"I don't see that in your notes." — do not elaborate further.

Context:
{context}

Question: {question}
""")

# ── STEP 6: HELPER FUNCTION ──────────────────────────────────────────────────
# Converts list of retrieved Document objects into one plain text block
# so it can be inserted into the {context} slot in the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── STEP 7: BUILD THE CHAIN ──────────────────────────────────────────────────
# LCEL pipe syntax connects steps left to right:
# retriever finds chunks → format_docs converts to text → fills {context}
# RunnablePassthrough passes question through unchanged → fills {question}
# prompt formats everything into the final prompt
# llm sends to Claude and gets response
# StrOutputParser extracts plain text from Claude's response object
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── STEP 8: INTERACTIVE Q&A LOOP ────────────────────────────────────────────
print("\nRAG app ready. Ask questions about your course notes (type 'quit' to exit)\n")

while True:
    question = input("Your question: ")

    if question.lower() == "quit":
        break

    # Retrieve source chunks separately so we can display them
    source_docs = retriever.invoke(question)

    # Run the full chain to get Claude's answer
    answer = chain.invoke(question)

    print(f"\nAnswer: {answer}")

    # Show which pages were retrieved so you can verify accuracy
    print("\nSources used:")
    for doc in source_docs:
        print(f"  - {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})")

    print("\n" + "-"*50 + "\n")