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
Stored in Session (Conversational Memory)
```

---

## 🧠Conversational Memory 

The application supports multi-turn conversations by maintaining a rolling window of recent chat history. This allows the system to:
1. Understand follow-up questions
2. Preserve contextual continuity
3. Provide more natural, chat-like interactions
 
Memory Characteristics:
1. Session-based (Streamlit session state)
2. Automatically injected into prompt construction
3. Lightweight and fully local
4. Resets when the application restarts
5. This ensures coherent conversations rather than isolated question-answer pairs.

---

## 🛠 Tech Stack

1. Programming Language: Python
2. LLM: Llama 3 (via Ollama)
3. Framework: LangChain
4. Vector Database: FAISS
5. Embeddings: Sentence Transformers (MiniLM)
6. Transcript API: youtube-transcript-api
7. UI Framework: Streamlit
8. Runtime: Fully local (no external APIs or cloud dependency)

## Demo Video

The demo showcases:
1. Loading a YouTube video using its video ID
2. Asking questions about the video content
3. Asking follow-up questions using conversational memory

Demo Video:


