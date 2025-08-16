# YouTube Video Q&A with RAG 💬

This is an interactive Q&A application that allows you to "chat" with any YouTube video. It uses a Retrieval-Augmented Generation (RAG) pipeline to understand the video's content and answer your questions based on the transcript.

## Features

- **Interactive Q&A:** Ask questions about a YouTube video in natural language.
- **Advanced Retrieval:** Uses a sophisticated retrieval pipeline with a reranker (**Cross-Encoder**) for highly accurate context finding.
- **Fast Generation:** Powered by the incredibly fast **Groq** API with Llama 3 for near-instant answers.
- **Open-Source Embeddings:** Utilizes a local, open-source model from Hugging Face for text embeddings.
- **Simple UI:** Built with **Streamlit** for a clean and easy-to-use web interface.

## Tech Stack

- **Framework:** LangChain
- **UI:** Streamlit
- **LLM:** Groq (Llama 3 8B)
- **Embedding Model:** Custom hash-based embeddings (Groq-only)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Data Loader:** `yt-dlp`

> **Note:** Streamlit Cloud runs with an old SQLite version that's incompatible with ChromaDB. So, ChromaDB is replaced with FAISS for better performance and compatibility.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/youtube-qa-chatbot.git
cd youtube-qa-chatbot
```

### 2. Create a Python Virtual Environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `streamlit` - Web UI framework
- `langchain` - RAG framework
- `langchain-groq` - Groq LLM integration
- `langchain-community` - Community integrations
- `faiss-cpu` - Facebook AI Similarity Search vector database
- `yt-dlp` - YouTube transcript extraction
- `python-dotenv` - Environment variable management

### 4. Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Get your API key from the [GroqCloud Console](https://console.groq.com/keys).

3. Open the `.env` file and add your API key:
   ```
   GROQ_API_KEY="your_groq_api_key_here"
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1. Paste a YouTube video URL into the sidebar input field.
2. Click the **"Process Video"** button and wait for the processing to complete.
3. Once processed, you can ask questions about the video in the main input field.

## 🚀 Deploy to Streamlit Cloud (Free)

To make your app accessible to others:

1. **Visit** https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Select your repository:** `Oghenesuvwe-dev/NSKAI-RAG-Bootcamp-2025`
5. **Set main file:** `app.py`
6. **Add secrets:** Click "Advanced settings" → "Secrets" → Add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
7. **Click "Deploy"**

Your app will be live at: `https://your-app-name.streamlit.app`

## Project Structure

```
.
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── chain.py          # Creates the final RAG chain with the LLM
│   ├── chunker.py        # Splits documents into smaller chunks
│   ├── retriever.py      # Creates the retriever and reranker
│   ├── vectorstore.py    # Creates the ChromaDB vector store
│   └── youtubeloader.py  # Loads and cleans transcripts using yt-dlp
├── .env                  # Stores API keys (secret, not committed to git)
├── .env.example          # Example environment file
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## How It Works

The application follows a standard RAG pipeline:

1. **Ingestion:** The `youtubeloader` fetches the video transcript using `yt-dlp` and cleans it.
2. **Chunking:** The `chunker` splits the clean transcript into smaller, overlapping documents.
3. **Indexing:** The `vectorstore` helper uses custom hash-based embeddings to create numerical vectors for each chunk and stores them in FAISS vector database.
4. **Retrieval:** When a question is asked, the `retriever` finds the most relevant chunks using FAISS similarity search.
5. **Generation:** The top-ranked chunks and the original question are passed to the Groq LLM within a structured prompt, which then generates the final, grounded answer.