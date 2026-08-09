# Chat with YouTube Videos using Local LLMs (RAG)

A **Retrieval-Augmented Generation (RAG)** application that allows users to **chat with YouTube videos** using **Llama 3 running locally through Ollama**.

The application extracts English transcripts from YouTube videos, splits them into overlapping chunks, generates semantic embeddings using Sentence Transformers, and stores them in a FAISS vector database.

It also supports **session-based conversational memory and history-aware query rewriting** to handle multi-turn conversations and follow-up questions.

---

## Features

- Chat with YouTube videos using a YouTube Video ID
- Automatic English transcript extraction
- Transcript chunking with overlapping text
- Semantic search using Sentence Transformers and FAISS
- Local Llama 3 inference using Ollama
- Session-based conversational memory
- History-aware query rewriting for follow-up questions
- Retrieval using both the original and rewritten query when rewriting is successful
- Context-grounded answers based only on retrieved transcript content
- Interactive Streamlit chat interface
- Local LLM inference without an external LLM API

---

# Architecture

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

# Conversational Memory

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

# History-Aware Query Rewriting

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



# Retrieval Strategy

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

# Answer Generation

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

# Complete Pipeline

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

# Tech Stack

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

### Demo Video

https://drive.google.com/file/d/1np0YwixGpS5EqLwGm3XPX4HtdO8DBJTl/view?usp=sharing

### YouTube Video Used for the Demo

https://www.youtube.com/watch?v=9EqrUK7ghho

---

