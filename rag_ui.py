import streamlit as st
api_key = st.secrets["GROQ_API_KEY"]
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# -------------------------------
# Helper
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------
# Chain builder (cached)
# -------------------------------
@st.cache_resource
def load_rag_chain(pdf_path, groq_model, api_key):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Local embeddings (fast, no API key)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=None)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Groq LLM with updated model
        llm = ChatGroq(model=groq_model, temperature=0, api_key=api_key)
        
        template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain, retriever
    except Exception as e:
        st.error(f"Failed to load chain: {str(e)}")
        return None, None

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Fast PDF RAG with Groq", page_icon="⚡")
st.title("⚡ PDF RAG with Groq (Super Fast LLM)")
st.write("Upload a PDF and ask questions – Groq provides lightning‑fast answers.")

with st.sidebar:
    st.header("Configuration")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getbuffer())
            pdf_path = tmp.name
        st.success(f"Loaded: {uploaded_pdf.name}")
    else:
        pdf_path = None
        st.info("Please upload a PDF.")
    
    # Updated model list with supported models
    groq_model = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",   # recommended, latest Llama 3.3
            "llama3-70b-8192",           # Llama 3 70B
            "llama3-8b-8192",            # Llama 3 8B (faster)
            "gemma2-9b-it"               # Google Gemma 2 9B
        ]
    )
    
    # Option to override API key from .env, or use environment variable
    api_key_input = st.text_input("Groq API Key (optional, leave blank to use .env)", type="password")
    st.caption("If left blank, the key from `.env` file will be used.")
    
    if st.button("🔄 Clear cache"):
        st.cache_resource.clear()
        st.rerun()

# Get API key: either from input or from environment
api_key = api_key_input if api_key_input else os.getenv("GROQ_API_KEY")

if pdf_path and api_key:
    query = st.text_area("Ask a question about the PDF:")
    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving from PDF and generating with Groq..."):
                rag_chain, retriever = load_rag_chain(pdf_path, groq_model, api_key)
                if rag_chain is None:
                    st.stop()
                try:
                    answer = rag_chain.invoke(query)
                    source_docs = retriever.invoke(query)
                    st.markdown("### Answer")
                    st.write(answer)
                    if source_docs:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(source_docs, 1):
                                st.write(f"{i}. {doc.page_content.strip()}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
elif not api_key:
    st.warning("Please provide a Groq API key (either in .env or in the sidebar).")
else:
    st.info("👈 Upload a PDF to start.")