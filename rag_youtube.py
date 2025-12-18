#Youtube_transcript
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

#LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)


#get transcript of video by video id
def get_transcript(video_id: str) -> str:
    """
    Fetch English transcript text for a YouTube video ID
    Compatible with youtube-transcript-api v1.2.3
    """
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])

        # v1.2.3 returns objects, not dicts
        return " ".join(item.text for item in transcript)

    except (TranscriptsDisabled, NoTranscriptFound):
        return ""


#build vector store
def build_vectorstore(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)


#format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#Build RAG Chain
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = Ollama(
        model="llama3",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
Use the chat history to understand follow-up questions.
If the context is insufficient, say "I don't know."

Chat History:
{chat_history}

Context:
{context}

Question: {question}
""",
        input_variables=["chat_history", "context", "question"],
    )

    chain = (
        RunnableParallel({
            # 🔑 IMPORTANT FIX: pass ONLY question to retriever
            "context": RunnableLambda(lambda x: x["question"])
                       | retriever
                       | RunnableLambda(format_docs),

            # Pass-through fields
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
