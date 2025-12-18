import streamlit as st
from rag_youtube import get_transcript, build_vectorstore, build_chain

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="YouTube Video Chat",
    page_icon="🎥",
    layout="wide"
)

# -----------------------
# Session State
# -----------------------
if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False


# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("🎬 Load YouTube Video")

    video_id = st.text_input(
        "YouTube Video ID",
        placeholder="Gfr50f6ZBvo"
    )

    load_btn = st.button("📥 Load Video")

    st.markdown("---")
    st.markdown(
        """
        **How it works**
        1. Fetch transcript
        2. Create embeddings
        3. Store in FAISS
        4. Chat using Llama 3
        """
    )

# -----------------------
# Main Header
# -----------------------
st.title("🎥 Chat with YouTube Video")
st.caption("Open-source RAG app using Llama 3, FAISS, and LangChain")

# -----------------------
# Load Video Logic
# -----------------------
if load_btn:
    st.session_state.messages = []
    st.session_state.chain = None
    st.session_state.video_loaded = False

    with st.spinner("📄 Fetching transcript..."):
        transcript = get_transcript(video_id)

    if not transcript:
        st.error("❌ No captions available for this video.")
    else:
        with st.spinner("🧠 Building vector store..."):
            vectorstore = build_vectorstore(transcript)
            st.session_state.chain = build_chain(vectorstore)

        st.session_state.video_loaded = True
        st.success("✅ Video loaded successfully! Ask your questions below.")

# -----------------------
# Chat Interface
# -----------------------
if st.session_state.video_loaded:

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the video...")

    if user_input:
        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = st.session_state.chain.invoke(user_input)
                st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👈 Enter a YouTube video ID in the sidebar to get started.")
