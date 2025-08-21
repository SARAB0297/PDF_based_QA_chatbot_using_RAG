# 📚 Conversational RAG-Based PDF Q&A System

An interactive **Retrieval-Augmented Generation (RAG)** application that enables users to upload one or more PDF files and engage in a **dynamic, session-aware Q&A conversation**.  
Built with **LangChain, HuggingFace Embeddings, Groq's Gemma2-9B-IT**, and a clean **Streamlit interface**, the system delivers fast, accurate, and context-aware answers from PDF content.

---

## 🚀 Features
- 🔍 **RAG Architecture** – Combines semantic document retrieval with LLM-based generation.  
- 🧠 **Session-Aware Chat** – Maintains multi-turn dialogue with memory for context-rich responses.  
- 📄 **Multi-PDF Upload** – Supports simultaneous PDF uploads with efficient parsing via `PyPDFLoader`.  
- 💬 **Interactive UI** – Built with Streamlit for seamless, real-time interaction.  
- ⚡ **Groq-Powered LLM** – Integrates **Gemma2-9B-IT** for fast and intelligent response generation.  
- 🔗 **Semantic Embeddings** – Utilizes HuggingFace **all-MiniLM-L6-v2** for document understanding.  
- ⚙️ **Custom Vector Store** – Stores chunks efficiently in **ChromaDB** with optimized retrieval.  

---

## 🛠️ Tech Stack
| Component         | Tool/Library                         |
|-------------------|---------------------------------------|
| Frontend          | Streamlit                            |
| Backend           | Python, LangChain                    |
| LLM               | Groq Gemma2-9B-IT (via Groq API)     |
| Embeddings        | HuggingFace - all-MiniLM-L6-v2       |
| Document Loader   | PyPDFLoader                          |
| Vector Store      | ChromaDB                             |
| Text Splitting    | RecursiveCharacterTextSplitter       |
| Session History   | LangChain ChatMessageHistory         |

---

## 📝 How It Works
1. User uploads one or more PDF documents.  
2. Content is split into semantic chunks using `RecursiveCharacterTextSplitter`.  
3. Embeddings are generated via HuggingFace and stored in **ChromaDB**.  
4. A **RAG pipeline** retrieves relevant chunks and generates concise answers using **Gemma2-9B-IT**.  
5. Multi-turn conversation is enabled with **session-aware memory**.  

---

## 🔮 Future Enhancements
- Support multiple embedding models (e.g., InstructorXL, OpenAI).  
- Add authentication / user login for secure sessions.  
- Deploy on cloud with **persistent vector storage**.  
- Integrate PDF preview + source highlighting for transparency.  

## 🧑‍💻 Author
**Sarabjeet Singh**  
---
