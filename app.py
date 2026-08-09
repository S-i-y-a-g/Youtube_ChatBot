import streamlit as st

from rag_youtube import (
    get_transcript,
    build_vectorstore,
    build_chain
)




st.set_page_config(
    page_title="YouTube Video Chat",
    layout="wide"
)




if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

if "video_id" not in st.session_state:
    st.session_state.video_id = None




def format_chat_history(
    messages,
    max_messages=8
):
    """
    Format recent conversation history
    for the LLM.

    The current user question is NOT included.
    """

    history = messages[-max_messages:]

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )




with st.sidebar:

    st.title("Load YouTube Video")

    video_id = st.text_input(
        "YouTube Video ID",
        placeholder="Gfr50f6ZBvo"
    )

    load_btn = st.button(
        "Load Video",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown(
        """
        ### How it works

        1. Fetch YouTube transcript
        2. Split transcript into chunks
        3. Generate embeddings
        4. Build FAISS vector store
        5. Rewrite contextual follow-up questions
        6. Retrieve relevant transcript chunks
        7. Generate answers using Llama 3
        """
    )

   

    if st.session_state.video_loaded:

        st.markdown("---")

        st.markdown("### Current Video")

        st.code(
            st.session_state.video_id
        )

        if st.button(
            "Clear Conversation",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()




st.title("Chat with YouTube Video")

st.caption(
    "History-aware conversational RAG "
    "using Llama 3 + FAISS"
)



if load_btn:

  

    if not video_id.strip():

        st.error(
            "Please enter a YouTube video ID."
        )

    else:

        

        st.session_state.messages = []

        st.session_state.chain = None

        st.session_state.video_loaded = False

        st.session_state.video_id = None

        

        with st.spinner(
            "Fetching YouTube transcript..."
        ):

            transcript = get_transcript(
                video_id.strip()
            )

       

        if not transcript:

            st.error(
                "No English captions were found "
                "for this video."
            )

        else:

           

            with st.spinner(
                "Creating embeddings and "
                "building FAISS index..."
            ):

                vectorstore = (
                    build_vectorstore(
                        transcript
                    )
                )

           

            st.session_state.chain = (
                build_chain(
                    vectorstore
                )
            )

            st.session_state.video_loaded = True

            st.session_state.video_id = (
                video_id.strip()
            )

            st.success(
                "Video loaded successfully! "
                "You can start chatting."
            )

            st.rerun()




if st.session_state.video_loaded:

   

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

   
    user_input = st.chat_input(
        "Ask a question about the video..."
    )

    if user_input:

        

        chat_history = (
            format_chat_history(
                st.session_state.messages
            )
        )

        

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_input
            )

        

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the transcript..."
            ):

                try:

                    result = (
                        st.session_state.chain(
                            question=user_input,
                            chat_history=chat_history
                        )
                    )

                    answer = result["answer"]

                except Exception as e:

                    st.error(
                        "An error occurred while "
                        "generating the answer."
                    )

                    st.exception(e)

                    answer = None

            

            if answer:

                st.markdown(
                    answer
                )

                

                with st.expander(
                    "View retrieval details"
                ):

                    
                    if result["was_rewritten"]:

                        st.success(
                            "History-aware query "
                            "rewriting was used."
                        )

                    else:

                        st.info(
                            "Original user query "
                            "was used for retrieval."
                        )

                    # Actual search query
                    st.write(
                        "**Search query used:**"
                    )

                    st.code(
                        result["search_query"]
                    )

                    # Retrieved transcript chunks
                    st.write(
                        "**Retrieved transcript chunks:**"
                    )

                    documents = result[
                        "retrieved_documents"
                    ]

                    if documents:

                        for i, document in enumerate(
                            documents,
                            start=1
                        ):

                            st.markdown(
                                f"**Chunk {i}**"
                            )

                            st.write(
                                document.page_content
                            )

                            if i != len(documents):

                                st.markdown("---")

                    else:

                        st.write(
                            "No transcript chunks "
                            "were retrieved."
                        )

               

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

else:

    

    st.info(
        "Enter a YouTube video ID in the "
        "sidebar to get started."
    )

    
