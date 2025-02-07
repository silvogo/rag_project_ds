import requests
import streamlit as st

def get_api_response(question, session_id, model="gpt-4o-mini"):
    # define headers
    headers = {
        "accept" : "application/json",
        "Content-Type": "application/json"
    }

    data = {
        "question": question,
        "model": model
    }

    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json= data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed! Status code {response.status_code}: {response.text}")
            return None

    except Exception as e:
        st.error(f"Some error occured when calling the chat API: {e}")
        return None

def upload_file (file):
    try:
        files = {
            "file": (file.name, file, file.type)
        }
        response = requests.post("http://localhost:8000/upload-doc",  files=files )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload the file! Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Some error occured when uploading the file: {e}")
        return None


