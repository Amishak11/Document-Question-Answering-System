# 📄 Document Question Answering System (NLP + LLM)

An intelligent document-based Question Answering system that leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to extract meaningful answers from uploaded documents. This project enables users to ask questions and receive accurate, context-aware responses directly from the document content.

---

## 🚀 Features

- 📑 Extracts and processes text from documents  
- ✂️ Performs text preprocessing & document chunking  
- 🔍 Uses vector embeddings for semantic search  
- 📊 Efficient similarity search using FAISS / ChromaDB  
- 🤖 Integrates Gemini AI (LLM) for generating answers  
- ⚡ Fast and context-aware responses  

---

## 🛠️ Tech Stack

- Python  
- NLP (NLTK / preprocessing techniques)  
- FAISS / ChromaDB (Vector Database)  
- Gemini AI API  
- Flask / Eel (optional for UI/backend)  

---

## ⚙️ How It Works

1. Upload a document (PDF/Text)  
2. Text is extracted and preprocessed  
3. Document is divided into smaller chunks  
4. Chunks are converted into vector embeddings  
5. Stored in FAISS / ChromaDB  
6. User query is converted into embedding  
7. System retrieves most relevant chunks  
8. Gemini AI generates the final answer  

---

## 📂 Project Structure

```
project/
│── data/               # Input documents
│── embeddings/         # Stored vector data
│── preprocessing.py    # Text cleaning & chunking
│── vector_store.py     # FAISS/ChromaDB setup
│── main.py             # Main application logic
│── requirements.txt    # Dependencies
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/document-qa-system.git

# Navigate to project folder
cd document-qa-system

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

- Upload your document  
- Ask questions in natural language  
- Get accurate answers instantly  

---

## 📌 Example

**Input:**  
"What is the main topic of the document?"

**Output:**  
"The document primarily discusses..."

---

## 🎯 Future Improvements

- Support for multiple file formats (DOCX, PPT)  
- Web-based UI for better interaction  
- Improved ranking using advanced embeddings  
- Multi-document querying  

---

## 🤝 Contribution

Feel free to fork this repository and contribute!  
Pull requests are welcome.

---

## 📜 License

This project is open-source and available under the MIT License.
