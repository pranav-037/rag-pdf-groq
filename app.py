import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# ----- Configuration -----
PDF_FILE = "document.pdf"
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "llama3-70b-8192", "gemma2-9b-it"

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

# ----- Check PDF -----
if not os.path.exists(PDF_FILE):
    print(f"Error: PDF file '{PDF_FILE}' not found.")
    exit(1)

# ----- Load PDF -----
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()

# ----- Split -----
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# ----- Local embeddings (fast, free) -----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----- Groq LLM (very fast) -----
llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

# ----- Prompt -----
template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# ----- RAG chain -----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ----- Ask -----
query = "What is the main topic of this document?"
result = rag_chain.invoke(query)
print("Answer:", result)

# ----- Show sources -----
source_docs = retriever.invoke(query)
print("\nSources:")
for i, doc in enumerate(source_docs, 1):
    print(f"{i}. {doc.page_content[:200]}...")