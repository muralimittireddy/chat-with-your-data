# Chat with your data (Local RAG Project)

It uses a Retrieval-Augmented Generation (RAG) pipeline to allow you to upload a PDF and ask questions about its content. The AI's knowledge is strictly limited to the document you provide.

This project is a fully functional Minimum Viable Product (MVP) built with a Python Flask backend, a modern web UI, and a 100% local, private LLM (Ollama).

## Project Features (Current)

- **Simple Web UI:** A clean interface to upload a single PDF.

- **Dedicated Chat:** The entire chat session is dedicated to the uploaded document.

- **Real-Time RAG:** Implements the full RAG pipeline (chunk, embed, retrieve, generate) in real-time on upload.

- **Streaming Responses:** The AI's answers are streamed to the UI word-by-word.

- **100% Private:** Uses a local embedding model and a local LLM (Ollama), so your documents and chats never leave your server.
