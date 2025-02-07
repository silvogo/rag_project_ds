import streamlit as st
from click import prompt

from frontend.api_utils import get_api_response, upload_file


def display_chat():

    for msg in st.session_state.messages:
        # creates a chat bubble
        with st.chat_message(msg["role"]) :
            # displays the actual message text inside the bubble
            st.markdown(msg["content"])

    # create prompt chat
    if prompt := st.chat_input("Question:"):
        # stores the message written by the user
        st.session_state.messages.append({"role": "user", "content": prompt})

        # creates a chat bubble for the user
        with st.chat_message("user"):
            # displays the message inside the chat bubble
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id)

            if response:
                answer = response["answer"]

                # store the assistant message
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Display the assistant message in chat UI
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                st.error("Error when getting response from the API.")


def display_bar_upload_doc():
    # Sidebar to upload documents
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a type of file", type=['csv', 'xls'])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading file..."):
                upload_response = upload_file(uploaded_file)