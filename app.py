from flask import Flask, request, jsonify, Response, render_template
import pypdf
import requests
import json
from io import BytesIO

# Import RAG-specific libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss # Facebook's fast vector search library
import numpy as np

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Global "In-Memory" Database ---
vector_store = None
chat_history = []
text_chunks = [] # Store original text chunks
embedding_model = None

# --- 3. Helper Functions for RAG ---

def load_embedding_model():
    """Loads the sentence-transformer model into memory."""
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        # We use a free, open-source model that runs locally
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")

def process_pdf(pdf_file):
    """Reads PDF, splits into chunks, creates embeddings, and builds a FAISS vector store."""
    global vector_store, text_chunks, chat_history

    try:
        print("Reading PDF...")
        pdf_reader = pypdf.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() or ""

        if not full_text:
            print("Warning: No text extracted from PDF.")
            return False

        print(f"Extracted {len(full_text)} characters.")

        # 2. Chunk the text
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        text_chunks = text_splitter.split_text(full_text)
        print(f"Created {len(text_chunks)} text chunks.")

        if not text_chunks:
            print("Error: No text chunks created.")
            return False

        # 3. Create Embeddings
        print("Creating embeddings...")
        load_embedding_model() # Ensure model is loaded
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
        print(f"Created {len(embeddings)} embeddings.")

        # 4. Build FAISS Vector Store
        print("Building FAISS vector store...")
        dimension = embeddings.shape[1]
        vector_store = faiss.IndexFlatL2(dimension)
        vector_store.add(np.array(embeddings).astype('float32'))
        print("FAISS vector store built successfully.")

        # Reset chat history for the new PDF
        chat_history = []

        return True
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False

def retrieve_context(query, top_k=3):
    """Retrieves the top-k relevant text chunks from the vector store."""
    if vector_store is None:
        return ""

    print(f"Retrieving context for query: {query}")
    # 1. Create embedding for the query
    query_embedding = embedding_model.encode([query])

    # 2. Search FAISS for similar vectors
    # D = distances, I = indices
    distances, indices = vector_store.search(np.array(query_embedding).astype('float32'), k=top_k)

    # 3. Get the original text chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n\n---\n\n".join(relevant_chunks)
    print(f"Retrieved context: {context[:200]}...")
    return context

def build_ollama_messages(query, context):
    """Builds the message history for the Ollama chat endpoint."""
    global chat_history

    # 1. Add the user's new query to history
    chat_history.append({"role": "user", "content": query})

    # 2. Create the system prompt
    system_prompt = f"""
    You are a helpful AI assistant. Your knowledge is strictly limited to the following context.
    Answer the user's question based *only* on the provided context.
    If the answer is not found in the context, say "I'm sorry, I don't have that information in the document."
    Do not use any external knowledge.

    CONTEXT:
    ---
    {context}
    ---
    """

    # 3. Build the final message list
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add recent history (e.g., last 4 messages) to maintain conversation flow
    messages.extend(chat_history[-4:]) # Add user query + last 3 messages

    return messages

def stream_ollama_response(messages):
    """Streams the response from Ollama and updates chat history."""
    global chat_history

    ollama_url = 'http://localhost:11434/api/chat'
    payload = {
        "model": "gemma:2b", # This must match the model you pulled
        "messages": messages,
        "stream": True
    }

    try:
        response = requests.post(ollama_url, json=payload, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        full_ai_response = ""

        # This loops through the streaming response line by line
        for line in response.iter_lines():
            if line:
                try:
                    # Each line is a JSON object
                    chunk = json.loads(line.decode('utf-8'))

                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        full_ai_response += content
                        yield content # This is what streams to the frontend

                    if chunk.get('done', False):
                        # The stream is finished
                        # Add the full AI response to our history
                        chat_history.append({"role": "assistant", "content": full_ai_response})
                        print(f"AI Response: {full_ai_response}")

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line: {line}")
                except Exception as e:
                    print(f"Error processing stream chunk: {e}")

    except requests.exceptions.ConnectionError:
        error_msg = "Error: Could not connect to Ollama. Please ensure Ollama is running."
        print(error_msg)
        chat_history.append({"role": "assistant", "content": error_msg})
        yield error_msg
    except Exception as e:
        error_msg = f"An error occurred with Ollama: {e}"
        print(error_msg)
        chat_history.append({"role": "assistant", "content": error_msg})
        yield error_msg

# --- 4. Define Flask API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handles PDF upload, processing, and vector store creation."""
    if 'pdf-upload' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf-upload']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            # Read file into a BytesIO object
            pdf_bytes = BytesIO(file.read())

            # Process the PDF
            success = process_pdf(pdf_bytes)

            if success:
                return jsonify({"message": f"Successfully processed {file.filename}"}), 200
            else:
                return jsonify({"error": "Failed to extract text from PDF."}), 500

        except Exception as e:
            print(f"Error in /upload: {e}")
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat query, RAG, and streams the response."""
    data = request.json
    user_query = data.get('message')

    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    if vector_store is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    # 1. Retrieve context
    context = retrieve_context(user_query)

    # 2. Build the prompt
    messages = build_ollama_messages(user_query, context)

    # 3. Stream the response
    # We return a Response object with the generator function
    return Response(stream_ollama_response(messages), mimetype='text/plain')

# --- 5. Run the App ---
if __name__ == '__main__':
    load_embedding_model() # Load the model on startup
    app.run(host='0.0.0.0', debug=True, port=5000)