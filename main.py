import eel
import os
import json
import base64
import io
import PyPDF2
import docx
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize

# --- 1. Project Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_gemini_api_key_here":
    print("WARNING: GEMINI_API_KEY is not set.")

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing client: {e}")
    client = None

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ChromaDB Initialization
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print("ChromaDB initialized successfully.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    chroma_client = None

# --- 2. History Storage ---
HISTORY_FILE = "chat_history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

@eel.expose
def get_all_chats():
    return load_history()

@eel.expose
def save_chat_message(session_id, role, text):
    history = load_history()
    if session_id not in history:
        history[session_id] = {
            "title": text[:30] + "..." if len(text) > 30 else text,
            "timestamp": datetime.now().isoformat(),
            "messages": []
        }
    history[session_id]["messages"].append({
        "role": role,
        "text": text,
        "timestamp": datetime.now().isoformat()
    })
    save_history(history)

@eel.expose
def delete_chat(session_id):
    history = load_history()
    if session_id in history:
        del history[session_id]
        save_history(history)
    return True

# --- 3. Model Setup ---
model_id = 'gemini-flash-latest'

current_date = datetime.now().strftime("%B %d, %Y")
system_prompt = f"You are Gemini Assistant, an incredibly advanced AI. The current date is {current_date}."

advanced_config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    tools=[{"google_search": {}}]
)
fallback_config = types.GenerateContentConfig(
    system_instruction=system_prompt
)

# Active chat sessions maintained here
active_sessions = {}

def get_or_create_session(session_id, force_model=None):
    use_model = force_model if force_model else model_id
    
    if session_id not in active_sessions or force_model:
        hist = load_history().get(session_id, {})
        genai_hist = []
        for msg in hist.get("messages", []):
            role = "user" if msg['role'] == 'user' else "model"
            genai_hist.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['text'])]))
            
        try:
            active_sessions[session_id] = client.chats.create(model=use_model, config=advanced_config, history=genai_hist)
        except Exception:
            active_sessions[session_id] = client.chats.create(model=use_model, config=fallback_config, history=genai_hist)
            
    return active_sessions[session_id], use_model

def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        print(f"[{len(tokens)} tokens] User Query Tokenized (NLP Pipeline): {tokens}")
        return text
    except Exception as e:
        print(f"Error in NLP pipeline: {e}")
        return text

# --- 3.5 Document Processing (RAG) ---
@eel.expose
def process_document(session_id, filename, file_data_base64):
    if not chroma_client:
        return "Error: Vector DB is not available."
    
    try:
        file_bytes = base64.b64decode(file_data_base64)
        text_content = ""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text: text_content += text + "\n"
        elif ext == '.docx':
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text_content += para.text + "\n"
        elif ext == '.txt':
            text_content = file_bytes.decode('utf-8')
        else:
            return f"Unsupported file type: {ext}"
            
        if not text_content.strip():
            return "Could not extract text from document."
            
        # NLP Pipeline: Text Preprocessing and Chunking
        print(f"Applying NLP chunking to {filename}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, 
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        
        collection_name = f"session_{session_id.replace('-', '')}"
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"Successfully processed '{filename}' into {len(chunks)} chunks."
    except Exception as e:
        print(f"Error processing document: {e}")
        return f"Error: {str(e)}"

# --- 4. Main Generation Engine ---
@eel.expose
def generate_response(session_id, user_input):
    if not client:
        return "Error: Gemini client not initialized. Check your API key."
        
    print(f"Received via session [{session_id}]: {user_input}")
    cleaned_input = preprocess_text(user_input)
    
    # RAG: Retrieve context if exists
    if chroma_client:
        collection_name = f"session_{session_id.replace('-', '')}"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            if collection.count() > 0:
                print(f"Querying vector DB for context...")
                results = collection.query(query_texts=[cleaned_input], n_results=3)
                if results['documents'] and results['documents'][0]:
                    retrieved_chunks = results['documents'][0]
                    context_str = "\n\n---\n\n".join(retrieved_chunks)
                    context_prompt = f"Use the following retrieved document context to help answer the user's question. If the answer is not in the context, you can still answer normally from your knowledge (but prefer the document).\n\nRetrieved Context:\n{context_str}\n\nUser Question:\n"
                    cleaned_input = context_prompt + cleaned_input
                    print(f"Appended {len(retrieved_chunks)} context chunks from Vector DB.")
        except Exception as e:
            # InvalidCollectionException is thrown if it does not exist yet; we just ignore it
            pass
    
    chat_session, current_model = get_or_create_session(session_id)
    
    models_to_try = [current_model, 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash-lite', 'gemini-pro-latest']
    last_error = ""
    
    for try_model in models_to_try:
        try:
            if try_model != current_model:
                print(f"Falling back to model: {try_model}")
                chat_session, current_model = get_or_create_session(session_id, force_model=try_model)
                
            response = chat_session.send_message(cleaned_input)
            return response.text
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            print(f"LLM API Error with {try_model}: {error_str}")
            
            if "RESOURCE_EXHAUSTED" in error_str or "NOT_FOUND" in error_str or "404" in error_str or "not supported" in error_str or "UNAVAILABLE" in error_str or "503" in error_str:
                continue
            else:
                return f"Error connecting to Gemini API: {error_str}"
                
    return f"Error: No available Gemini models in your region/quota. Last error: {last_error}"

# --- 5. Application Launch ---
if __name__ == '__main__':
    eel.init('.')
    try:
        print("Starting Enhanced AI Chatbot UI...")
        eel.start('index.html', mode='chrome', size=(1100, 850), port=0)
    except (SystemExit, MemoryError, KeyboardInterrupt):
        pass
