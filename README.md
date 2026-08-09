# 🎥 Chat with YouTube Videos using Local LLMs (RAG)

A **Retrieval-Augmented Generation (RAG)** application that allows users to **chat with YouTube videos** using **Llama 3 running locally through Ollama**.

The application extracts English transcripts from YouTube videos, splits them into overlapping chunks, generates semantic embeddings using Sentence Transformers, and stores them in a FAISS vector database.

It also supports **session-based conversational memory and history-aware query rewriting** to handle multi-turn conversations and follow-up questions.

---

## ✨ Features

- 🎥 Chat with YouTube videos using a YouTube Video ID
- 📝 Automatic English transcript extraction
- ✂️ Transcript chunking with overlapping text
- 🔎 Semantic search using Sentence Transformers and FAISS
- 🤖 Local Llama 3 inference using Ollama
- 💬 Session-based conversational memory
- 🧠 History-aware query rewriting for follow-up questions
- 🔍 Retrieval using both the original and rewritten query when rewriting is successful
- 📚 Context-grounded answers based only on retrieved transcript content
- 🖥️ Interactive Streamlit chat interface
- 🔒 Local LLM inference without an external LLM API

---

# 🏗️ Architecture

The application follows a Retrieval-Augmented Generation pipeline with conversational memory.

## 1. Transcript Extraction

The user provides a YouTube video ID.

The application uses:

`youtube-transcript-api`

to fetch the video's English transcript.

The transcript snippets are combined into a single text document.

---

## 2. Text Chunking

The transcript is split into smaller overlapping chunks using:

`RecursiveCharacterTextSplitter`

Current configuration:

- Chunk size: `1000`
- Chunk overlap: `200`

The overlap helps preserve context between neighboring transcript sections.

---

## 3. Embedding Generation

Each transcript chunk is converted into a dense vector representation using:

`sentence-transformers/all-MiniLM-L6-v2`

These embeddings allow the system to perform semantic similarity search.

---

## 4. FAISS Vector Store

The generated embeddings are stored in a:

`FAISS`

vector database.

FAISS is used to efficiently retrieve transcript chunks that are semantically related to the user's question.

---

# 🧠 Conversational Memory

The application maintains recent conversation history using:

`st.session_state.messages`

The conversation history contains previous user and assistant messages.

A rolling window of the most recent **8 messages** is passed to the RAG pipeline.

For example:

```text
User:
What does the speaker say about confidence?

Assistant:
The speaker says that confidence comes from practice.

User:
Why does she think it is important?
```

The previous messages allow the system to understand references such as:

- it
- this
- that
- he
- she
- they
- the previous point
- the second approach

### Memory Characteristics

- Session-based
- Stored using Streamlit `session_state`
- Limited to recent conversation messages
- Used for query rewriting
- Used during answer generation
- Fully local
- Not persisted to a database
- Resets when the Streamlit session is restarted

> **Note:** FAISS stores the YouTube transcript embeddings. It is not used to store conversational memory.

---

# 🔄 History-Aware Query Rewriting

When conversation history exists, the system attempts to rewrite the current question into a standalone search query.

The query rewriting process uses:

```text
Conversation History
        +
Current User Question
        ↓
      Llama 3
        ↓
Standalone Search Query
```

The rewriting prompt instructs Llama 3 to:

- Preserve important keywords
- Preserve technical terms
- Preserve names, dates, numbers, and topics
- Resolve references such as "it", "this", "that", "he", and "she"
- Avoid making the question more general
- Avoid replacing specific topics with generic phrases
- Return a standalone search query

### Example

Previous conversation:

```text
User:
What is reinforcement learning?

Assistant:
Reinforcement learning is a machine learning approach...
```

Follow-up question:

```text
Why is it useful?
```

The system may rewrite the query as:

```text
Why is reinforcement learning useful?
```

The rewritten query is then used as an additional retrieval query.

---

# 🛡️ Query Validation

Because an LLM can sometimes produce an incorrect or overly generic rewrite, the application validates the rewritten query before using it.

The validation checks whether:

- The rewritten query is empty
- The rewritten query is too short
- Important content words from the original question were removed
- The rewritten query became overly generic

For example:

```text
Original Question:
Has she mentioned anything about confidence?

Bad Rewrite:
What does the video discuss?
```

The rewritten query can be rejected because it loses the important topic:

```text
confidence
```

If validation fails, the application falls back to the original user question.

This prevents a poor query rewrite from completely changing the user's search intent.

---

# 🔍 Retrieval Strategy

The system always performs retrieval using the **original user question**.

If the query is successfully rewritten, it also performs retrieval using the **rewritten query**.

The retrieved documents from both searches are then combined and deduplicated.

```text
                    User Question
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Original Query        Rewritten Query
              │                     │
              ▼                     ▼
            FAISS                 FAISS
              │                     │
              └──────────┬──────────┘
                         ▼
                  Retrieved Chunks
                         │
                         ▼
                    Deduplication
                         │
                         ▼
                     Top 6 Chunks
```

If query rewriting is not used or fails validation:

```text
User Question
      ↓
    FAISS
      ↓
Retrieved Chunks
```

This approach ensures that the original question is always represented during retrieval.

---

# 🤖 Answer Generation

After retrieval, the system constructs a prompt containing:

- Recent conversation history
- Retrieved transcript context
- Original user question

The final answer is generated using:

**Llama 3 via Ollama**

The model is instructed to:

- Answer only using the provided transcript context
- Preserve the original question's intent
- Use conversation history to understand follow-up questions
- Avoid inventing information
- Avoid using outside knowledge

If the retrieved transcript does not contain enough information, the model is instructed to respond:

```text
I don't know based on the transcript.
```

---

# 🔄 Complete Pipeline

```text
                 User
                  │
                  ▼
        YouTube Video ID
                  │
                  ▼
      youtube-transcript-api
                  │
                  ▼
        English Transcript
                  │
                  ▼
  RecursiveCharacterTextSplitter
                  │
                  ▼
          Text Chunks
                  │
                  ▼
      Sentence Transformers
                  │
                  ▼
          FAISS Vector Store
                  │
                  │
                  ▼
          User Question
                  │
                  ├───────────────────┐
                  │                   │
                  ▼                   ▼
           Original Query       Chat History
                  │                   │
                  │                   ▼
                  │                Llama 3
                  │            Query Rewriting
                  │                   │
                  │                   ▼
                  │           Query Validation
                  │                   │
                  │             Valid Rewrite
                  │                   │
                  │                   ▼
                  │           Rewritten Query
                  │                   │
                  ▼                   ▼
                FAISS               FAISS
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    Retrieved Chunks
                            │
                            ▼
                       Deduplication
                            │
                            ▼
                       Top 6 Chunks
                            │
                            ▼
                    Prompt Construction
                            │
                 ┌──────────┼──────────┐
                 │          │          │
                 ▼          ▼          ▼
              History    Context    Question
                 │          │          │
                 └──────────┼──────────┘
                            ▼
                       Llama 3
                            │
                            ▼
                   Context-Aware Answer
                            │
                            ▼
                    Streamlit Chat UI
                            │
                            ▼
                 Session-based Memory
```

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Llama 3 |
| LLM Runtime | Ollama |
| Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Transcript Extraction | youtube-transcript-api |
| UI | Streamlit |

---

# ⚙️ Current Configuration

```text
LLM:
Llama 3

LLM Runtime:
Ollama

LLM Temperature:
0.1

Embedding Model:
sentence-transformers/all-MiniLM-L6-v2

Vector Store:
FAISS

Retriever:
Similarity Search

Initial Top-K:
4

Maximum Retrieved Chunks After Deduplication:
6

Chunk Size:
1000

Chunk Overlap:
200

Conversation History:
Last 8 messages
```

---

# 📁 Project Structure

```text
Youtube_ChatBot/
│
├── app.py
│
├── rag_youtube.py
│
├── requirements.txt
│
└── README.md
```

### `app.py`

Responsible for:

- Streamlit user interface
- YouTube video loading
- Session state
- Conversation history
- Chat interface
- Displaying retrieval information

### `rag_youtube.py`

Responsible for:

- YouTube transcript extraction
- Transcript chunking
- Embedding generation
- FAISS vector store creation
- Query rewriting
- Query validation
- Semantic retrieval
- Document deduplication
- Llama 3 answer generation

### `requirements.txt`

Contains the Python dependencies required to run the application.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Youtube_ChatBot
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Install Ollama on your system.

Then download Llama 3:

```bash
ollama pull llama3
```

You can verify the model installation using:

```bash
ollama run llama3
```

---

## 5. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 💬 Example Conversation

The application supports multi-turn questions.

### Example

```text
User:
What does the speaker say about confidence?

Assistant:
The speaker explains that confidence comes from
practice, clear structure, and calm delivery.

User:
Why does she think it is important?

Assistant:
According to the transcript, she considers
confidence important because...
```

The second question can use the previous conversation to construct a standalone retrieval query.

---

# 🔍 Retrieval Debugging

The application includes a **View retrieval details** section.

For each response, users can inspect:

### 1. Query Rewriting Status

The application displays whether history-aware query rewriting was used.

If rewriting succeeds:

```text
History-aware query rewriting was used.
```

If rewriting is not used or validation fails:

```text
Original user query was used for retrieval.
```

### 2. Search Query

The application displays the actual search query used by the retrieval pipeline.

This can be either:

- The original user question
- A validated rewritten question

### 3. Retrieved Transcript Chunks

The application displays the transcript chunks retrieved from FAISS.

This makes it easier to debug retrieval quality and understand the context provided to Llama 3.

---

# 🎬 Demo

The demo showcases:

1. Loading a YouTube video using its video ID
2. Fetching the English transcript
3. Creating embeddings
4. Building a FAISS vector store
5. Asking questions about the video
6. Asking contextual follow-up questions
7. Using conversational memory
8. Performing history-aware query rewriting
9. Retrieving relevant transcript chunks
10. Generating answers using Llama 3

### Demo Video

https://drive.google.com/file/d/1np0YwixGpS5EqLwGm3XPX4HtdO8DBJTl/view?usp=sharing

### YouTube Video Used for the Demo

https://www.youtube.com/watch?v=9EqrUK7ghho

---

# 🔐 Privacy and Local Inference

The LLM inference is performed locally using:

```text
Llama 3 + Ollama
```

No external LLM API is required for query rewriting or answer generation.

Conversation history is maintained in the Streamlit session and is not stored in a persistent external database.

The application does access YouTube to retrieve the requested transcript.

---

# 🚧 Current Limitations

- English transcripts are currently used.
- Conversation history is session-based.
- Conversations are not persistently stored.
- FAISS is rebuilt when a video is loaded.
- Answer quality depends on transcript quality.
- Retrieval is based on semantic similarity.
- Query rewriting depends on Llama 3 and is validated before use.
- Local Llama 3 inference can be slower depending on the user's hardware.
- The application currently processes one loaded YouTube video at a time.

---

# 🔮 Future Improvements

Potential improvements include:

- Persistent conversation storage
- FAISS index caching
- Timestamp-aware retrieval
- Showing timestamps for retrieved transcript sections
- Source citations for generated answers
- Support for multiple videos
- Multilingual transcript support
- Streaming Llama 3 responses
- Hybrid keyword + semantic retrieval
- Improved summarization using multiple transcript sections
- Support for directly entering YouTube URLs
- Conversation export

---

# 📌 Project Highlights

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Semantic search
- Vector databases
- Sentence embeddings
- Local LLM inference
- Conversational memory
- History-aware query rewriting
- Query validation
- Context-grounded generation
- Prompt engineering
- Document retrieval and deduplication
- Streamlit application development
