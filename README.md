# Chat with your data (Local RAG Project)

It uses a Retrieval-Augmented Generation (RAG) pipeline to allow you to upload a PDF and ask questions about its content. The AI's knowledge is strictly limited to the document you provide.

This project is a fully functional Minimum Viable Product (MVP) built with a Python Flask backend, a modern web UI, and a 100% local, private LLM (Ollama).

## Project Features (Current)

- **Simple Web UI:** A clean interface to upload a single PDF.

- **Dedicated Chat:** The entire chat session is dedicated to the uploaded document.

- **Real-Time RAG:** Implements the full RAG pipeline (chunk, embed, retrieve, generate) in real-time on upload.

- **Streaming Responses:** The AI's answers are streamed to the UI word-by-word.

- **100% Private:** Uses a local embedding model and a local LLM (Ollama), so your documents and chats never leave your server.

## Architecture

This application uses a client-server model:

- **Frontend (Client):** A single index.html file using Tailwind CSS that runs in the user's browser.

- **Backend (Server):** A Python Flask server that handles all the heavy lifting:

  - Serves the index.html UI.

  - Provides a /upload endpoint to process PDFs.

  - Provides a /chat endpoint to run the RAG pipeline.

- **LLM:** The Ollama service (running on localhost:11434) serves the generative model (gemma:2b).

## RAG Pipeline Workflow:

1. **Upload:** User uploads a PDF to the Flask backend.

2. **Process:** The server reads the text with pypdf, splits it into chunks with langchain-text-splitters, and creates vector embeddings for each chunk using sentence-transformers.

3. **Store:** These embeddings are stored in-memory in a FAISS vector store.

4. **Chat:**

    - User sends a question (e.g., "What is chapter 2 about?").

    - Flask retrieves the most relevant text chunks from the FAISS store.

    - Flask "augments" a prompt, combining the user's question with the retrieved chunks.

    - This final prompt is sent to Ollama (gemma:2b).

    - The answer is streamed back to the UI.

---

## ⚙️ Tech Stack

### **Backend**
- **Python 3**
- **Flask** — Web server
- **pypdf** — Extract text from PDFs
- **langchain-text-splitters** — Split large documents into chunks
- **sentence-transformers** — Generate text embeddings (e.g., `all-MiniLM-L6-v2`)
- **faiss-cpu** — In-memory vector search for fast retrieval
- **requests** — Communicate with the Ollama API

### **LLM**
- **Ollama** — Local model server
- **Gemma:2b** — Open-source LLM for text generation

### **Frontend**
- **HTML**
- **Tailwind CSS** — Styling
- **Vanilla JavaScript (Fetch API)** — UI logic & streaming responses

---

## 🚀 Setup & Installation

These steps assume deployment on a **Linux-based server** (e.g., GCP VM).

### **1. Clone the Repository**
```bash
git clone https://github.com/muralimittireddy/chat-with-your-data.git
cd chat-with-your-data
```
### **2. Create and activate a Python virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

### **4. Ollama Setup (The "Brain")**

- **Install Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- **Start the Ollama service:**
(This is often done automatically, but you can run it manually in the background).

```bash
sudo systemctl start ollama
```

- **Pull the LLM model:**
(The code is configured for gemma:2b, but you can change it in app.py).

```bash
ollama pull gemma:2b
```

## How to Run

  1. **Run the Flask Web Server:**
  (Make sure you are in your project folder with the venv activated).
  
  ```bash
  python app.py
  ```
  
  The server will start on http://0.0.0.0:5000.
  
  2. **Check that Ollama is running:**
  (In a separate terminal, if needed).
  
  ```bash
  ollama list
  ```
  
  Ensure gemma:2b is in the list.
  
  ## How to Access
  
  you can access the app from your local browser at:
  
  ```bash
  http://localhost:5000
  ```

## Future Plans

This is an MVP. The next steps are to take this project to a production level, focusing on:

- New Features: Handling multiple documents, saving/loading chat history.

- Optimization: Moving from an in-memory FAISS store to a persistent vector database (e.g., ChromaDB, Milvus).

- Production Deployment: Securing the app with a Load Balancer (HTTPS) and running the Flask app with a production server like Gunicorn.
