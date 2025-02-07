import sys
import os
# Add the project root (one level up) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st

from frontend.chat_ui import display_chat, display_bar_upload_doc

# Definining the title of the APP
st.title("RAG Customer Satisfaction Chatbot")

# initialize session state variables to not lose previous values
if "messages" not in st.session_state:
    # messages is a key in special dictionary (st.session_state) that can be accessed using st.session_state.messages
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display sidebar
display_bar_upload_doc()
# Display chat
display_chat()
