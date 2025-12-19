# Chat with YouTube Videos using Local LLMs (RAG)

A **Retrieval-Augmented Generation (RAG)** based application that allows users to **chat with YouTube videos** using **open-source LLMs running fully locally**.

The system extracts video transcripts using a YouTube video ID, performs **semantic search** over the transcript, and generates **context-aware answers** with **conversational memory**.

---

## Architecture

1. **Transcript Extraction**  
   Fetches video subtitles using `youtube-transcript-api`.

2. **Text Chunking**  
   Splits the transcript into overlapping chunks using LangChain text splitters.

3. **Embeddings Generation**  
   Converts text chunks into dense vector embeddings using Sentence Transformers.

4. **Vector Storage**  
   Stores embeddings in a FAISS vector database for fast similarity search.

5. **Retrieval (RAG)**  
   Retrieves the most relevant transcript chunks based on the user’s query.

6. **LLM Generation**  
   Generates answers using **Llama 3** via **Ollama**, guided by retrieved context.

7. **Conversational Memory**  
   Maintains recent chat history to support multi-turn, follow-up questions.

---

## 🔄 Pipeline

```text
User enters YouTube Video ID
        ↓
youtube-transcript-api (v1.2.3)
        ↓
Full Transcript (English captions)
        ↓
RecursiveCharacterTextSplitter
        ↓
Overlapping Text Chunks
        ↓
Sentence-Transformers Embeddings
        ↓
FAISS Vector Database
        ↓
────────────────────────────────
        ↓
User Question + Chat History
        ↓
FAISS Retriever (Similarity Search)
        ↓
Top-K Relevant Transcript Chunks
        ↓
Prompt Construction:
   • Chat History
   • Retrieved Context
   • Current Question
        ↓
Llama 3 via Ollama (Local Inference)
        ↓
Context-Aware Answer
        ↓
Streamlit Chat UI
        ↓
Stored in Session (Conversational Memory) ```text

---

## 


