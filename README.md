# 📄 PDF RAG with Groq (Fast, Local Embeddings)

A production‑ready Retrieval‑Augmented Generation (RAG) system that lets you upload a PDF, ask questions, and get lightning‑fast answers using **Groq’s LPU** for the LLM and **local Hugging Face embeddings** for retrieval.  
No OpenAI keys, no expensive GPU – just a Groq API key and your PDF.

## ✨ Features

- ✅ Upload any **PDF** – extracts text automatically  
- ✅ **Fast LLM** – uses Groq’s `llama-3.3-70b-versatile` (or `llama3-70b-8192`)  
- ✅ **Local embeddings** – `all-MiniLM-L6-v2` runs on your CPU, no extra cost  
- ✅ **Streamlit UI** – clean, interactive web interface  
- ✅ **Secure** – API key stored in `.env` or Streamlit secrets  
- ✅ **Deployable** – works on Streamlit Cloud, Hugging Face Spaces, or your own server  

## 🧠 How it works

1. You upload a PDF.  
2. The text is split into chunks.  
3. Local embeddings create a vector store (Chroma).  
4. Your question retrieves the most relevant chunks.  
5. Groq LLM answers **only from those chunks** – no hallucination.  

## 📦 Requirements

- Python 3.8+  
- A [Groq API key](https://console.groq.com) (free tier works)  
- Internet connection (to download the embedding model once)

## 🚀 Quick Start


### 1. Clone the repository

```bash
git clone https://github.com/pranav-037/rag-pdf-groq.git
cd rag-pdf-groq


2. Install dependencies

pip install -r requirements.txt

3. Set up your API key

Create a .env file in the project root:
echo "GROQ_API_KEY=your_actual_key_here" > .env

4. Run the app

streamlit run rag_ui.py


Your browser will open at http://localhost:8501. Upload a PDF and start asking!

🖥️ Project structure

.
├── rag_ui.py               # Streamlit UI (main entry)
├── app.py                  # CLI version (optional)
├── requirements.txt        # Python dependencies
├── .env                    # API key (gitignored)
└── README.md               # This file


⚙️ Configuration
Changing the Groq model
Edit the groq_model selectbox in rag_ui.py or modify the GROQ_MODEL variable in app.py.
Supported models (as of April 2025):

llama-3.3-70b-versatile (recommended)

llama3-70b-8192

llama3-8b-8192 (faster, slightly less accurate)

gemma2-9b-it


Adjusting chunk size

In rag_ui.py, change:

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

Larger chunk_size gives more context but slower retrieval. 500 works well for most PDFs.


Using a different embedding model
Replace HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") with any Sentence‑Transformer model. For example, "BAAI/bge-small-en" is also fast and good.

☁️ Deployment
Streamlit Community Cloud (easiest)
Push your code to a GitHub repository (make sure .env is not committed).

Go to Streamlit Cloud, click New app.

Select your repo, branch, and main file: rag_ui.py.

In Advanced settings → Secrets, add:
GROQ_API_KEY = "your-key-here"

Deploy – your app will be live at https://your-app.streamlit.app.