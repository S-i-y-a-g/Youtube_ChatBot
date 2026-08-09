

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound
)

# LangChain Imports
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate



# Get YouTube Transcript
def get_transcript(video_id: str) -> str:
    """
    Fetch English transcript text for a YouTube video.
    """

    try:

        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            video_id,
            languages=["en"]
        )

        return " ".join(
            item.text
            for item in transcript
        )

    except (
        TranscriptsDisabled,
        NoTranscriptFound
    ):

        return ""



# Build Vector Store
def build_vectorstore(transcript: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents(
        [transcript]
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        documents,
        embeddings
    )



# Format Documents
def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



# Extract Words
def get_content_words(text: str):
    """
    Extract meaningful words from a query.

    Used to check whether Llama's rewritten
    query has accidentally removed important
    information.
    """

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "would",
        "should",
        "will",
        "about",
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
        "them",
        "he",
        "she",
        "we",
        "you",
        "i",
        "in",
        "on",
        "of",
        "to",
        "for",
        "from",
        "with",
        "and",
        "or",
        "but",
        "be",
        "been",
        "being",
        "any",
        "anything",
        "something"
    }

    words = (
        text.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .split()
    )

    return {
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    }



# Validate Rewritten Query
def validate_rewritten_query(
    original_question: str,
    rewritten_question: str
):
    """
    Check whether the rewritten query has
    accidentally removed important information.

    Example:

    Original:
        Has she mentioned anything about confidence?

    Bad rewrite:
        What does the video discuss?

    Result:
        False
    """

    original_words = get_content_words(
        original_question
    )

    rewritten_words = get_content_words(
        rewritten_question
    )

    
    # Empty / suspicious rewrite
    if not rewritten_question:
        return False

    if len(rewritten_question.split()) < 3:
        return False

   

    if original_words:

        overlap = (
            original_words
            .intersection(rewritten_words)
        )

        overlap_ratio = (
            len(overlap)
            / len(original_words)
        )

       
        if (
            len(original_words) >= 2
            and overlap_ratio < 0.30
        ):
            return False

    

    generic_phrases = [
        "what does the video discuss",
        "what is the video about",
        "what is discussed in the video",
        "what does the speaker discuss",
        "what is the main topic"
    ]

    normalized = (
        rewritten_question
        .lower()
        .strip()
        .replace("?", "")
    )

    for phrase in generic_phrases:

        if normalized == phrase:
            return False

    return True




def rewrite_query(
    question: str,
    chat_history: str,
    llm
):
    """
    Rewrite only when necessary.

    The rewritten query MUST preserve the
    user's actual information need.
    """

    

    if not chat_history.strip():

        return question, False

    rewrite_prompt = PromptTemplate(
        template="""
You are a search query rewriting assistant
for a YouTube transcript question-answering system.

Your job is to convert a follow-up question into
a standalone search query ONLY when the question
depends on previous conversation context.

IMPORTANT RULES:

1. NEVER remove important keywords from the user's
   current question.

2. NEVER make the question more general.

3. NEVER replace a specific topic with
   "the video" or "the topic".

4. Preserve important nouns and concepts exactly.

5. Preserve names, technical terms, dates,
   numbers, topics and subjects.

6. Resolve pronouns such as:
   "it", "this", "that", "he", "she", "they"
   using the conversation history when possible.

7. If the question is already understandable
   on its own, return it unchanged.

8. DO NOT answer the question.

Examples:

Conversation:
User: What does the speaker say about confidence?
Assistant: The speaker discusses...

Question:
Has she mentioned anything else about confidence?

GOOD:
Has the speaker mentioned anything else about confidence?

BAD:
What does the video discuss?

---

Conversation:
User: What is reinforcement learning?
Assistant: ...

Question:
Why is it useful?

GOOD:
Why is reinforcement learning useful?

BAD:
What does the video discuss?

---

Conversation:
User: Who is the speaker?
Assistant: The speaker is Alice.

Question:
What does she say about AI?

GOOD:
What does Alice say about AI?

BAD:
What does the speaker discuss?

Conversation History:
{chat_history}

Current Question:
{question}

Standalone Search Query:
""",
        input_variables=[
            "chat_history",
            "question"
        ]
    )

    rewritten = llm.invoke(
        rewrite_prompt.format(
            chat_history=chat_history,
            question=question
        )
    )

    rewritten = rewritten.strip()

   

    is_valid = validate_rewritten_query(
        question,
        rewritten
    )

    if not is_valid:

        
        return question, False

    return rewritten, True




def deduplicate_documents(
    documents
):
    """
    Remove duplicate transcript chunks.
    """

    seen = set()
    unique_docs = []

    for doc in documents:

        content = doc.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_docs.append(
                doc
            )

    return unique_docs




def build_chain(vectorstore):

    

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

   

    llm = Ollama(
        model="llama3",
        temperature=0.1
    )

    

    answer_prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions
about a YouTube video transcript.

Your answer must be based ONLY on the
provided transcript context.

Use the conversation history to understand
references and follow-up questions.

IMPORTANT:

- Answer the user's ORIGINAL question.
- Do not answer a rewritten question if it
  changes the user's intent.
- Preserve the specific topic asked about.
- Do not invent information.
- Do not use outside knowledge.

If the transcript does not contain enough
information to answer the question, say:

"I don't know based on the transcript."

Conversation History:
{chat_history}

Retrieved Transcript Context:
{context}

Original User Question:
{question}

Answer:
""",
        input_variables=[
            "chat_history",
            "context",
            "question"
        ]
    )

    

    def rag_function(
        question: str,
        chat_history: str
    ):

       

        search_query, was_rewritten = (
            rewrite_query(
                question=question,
                chat_history=chat_history,
                llm=llm
            )
        )

        

        original_docs = retriever.invoke(
            question
        )

       

        if was_rewritten:

            rewritten_docs = retriever.invoke(
                search_query
            )

            all_docs = (
                original_docs
                + rewritten_docs
            )

        else:

            all_docs = original_docs

       

        docs = deduplicate_documents(
            all_docs
        )

        # Limit context size
        docs = docs[:6]

        

        context = format_docs(
            docs
        )

       
        final_prompt = answer_prompt.format(
            chat_history=chat_history,
            context=context,
            question=question
        )

        answer = llm.invoke(
            final_prompt
        )

        return {
            "answer": answer.strip(),

            "search_query": search_query,

            "was_rewritten": was_rewritten,

            "retrieved_documents": docs
        }

    return rag_function
