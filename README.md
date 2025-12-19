It is a RAG based Application which allows users to chat with Youtube videos using Open Source LLMs running fully locally.
The system extracts video transcripts by video id, performs semantic search, and generates context-aware answers with conversational memory.

## Architecture
1. Transcript Extraction
  Fetches subtitles using youtube-transcript-api.

2. Text Chunking
  Splits transcript into overlapping chunks using LangChain text splitters.

3. Embeddings
  Converts chunks into vector embeddings using Sentence Transformers.

4. Vector Storage
  Stores embeddings in a FAISS vector database.

5. Retrieval (RAG)
  Retrieves the most relevant transcript chunks for each query.

6. LLM Generation
  Generates answers using Llama 3 via Ollama, guided by retrieved context.

7. Conversational Memory
   Maintains recent chat history for follow-up questions.


