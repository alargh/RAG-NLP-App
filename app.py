import streamlit as st
from dotenv import load_dotenv

from setup_nltk import setup_nltk
setup_nltk()

from src.summarizer import summarize
from src.chunk_splitter import get_chunks
from src.document_loader import get_text_from_files
from src.vector_store import get_vectors
from src.conversation import get_conversational_chain
from src.keyword_search import analyze_concert_tour_document


def process_question(message):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        return "Please upload text files first to provide context for answering questions."

    response = st.session_state.conversation(message)
    return response


def main():
    st.set_page_config(
        page_title='ChatBot',
    )

    st.title("2025-2026 concert tours ChatBot", anchor=None)

    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("Document Analysis Settings")
        keyword_threshold = st.slider(
            "Keyword Matching Threshold",
            min_value=0,
            max_value=100,
            value=20,
            help="Minimum number of keyword matches required to consider document relevant, if not enough text still can passthrough if transformer considers it relevant"
        )

        st.subheader("Summarization Settings")
        summary_threshold = st.slider(
            "Summary Threshold Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Higher values produce shorter summaries, if set to 0.0 - it ingests full document"
        )

    text_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)

    if text_files:
        ingest_button = st.button("Ingest Files")
        if ingest_button:
            with st.spinner("Processing uploaded files..."):
                raw_text = get_text_from_files(text_files)

                is_relevant = analyze_concert_tour_document(raw_text, keyword_threshold)

                if not is_relevant:
                    st.error(
                        "The uploaded documents don't appear to be relevant concert tour documents. Please upload relevant files.")

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "I've uploaded some text files."
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "The uploaded documents don't appear to be relevant concert tour documents. Please upload relevant files."
                    })
                else:

                    summary = summarize(raw_text, thresh_ratio=summary_threshold)

                    orig_word_count = len(raw_text.split())
                    summary_word_count = len(summary.split())

                    text_chunks = get_chunks(summary)
                    st.sidebar.write(f"Created {len(text_chunks)} chunks")

                    vector = get_vectors(text_chunks, st.session_state.vector_store)

                    st.session_state.vector_store = vector
                    st.session_state.conversation, st.session_state.memory = get_conversational_chain(vector)

                    st.success(f"Files successfully ingested! Split into {len(text_chunks)} chunks.")

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "I've uploaded some text files. Can you summarize them for me?"
                    })

                    summary_response = (
                        f"Here's a summary of the concert tour documents you uploaded:\n\n{summary}\n\n"
                        f"*(Original: {orig_word_count} words | Summary: {summary_word_count} words)*\n\n"
                        f"You can now ask me questions about the content."
                    )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": summary_response
                    })

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the ingested text")

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            if st.session_state.conversation:
                response = process_question(question)

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
            else:
                response = "Please upload and ingest text files first to provide context for answering questions."

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })


if __name__ == '__main__':
    main()