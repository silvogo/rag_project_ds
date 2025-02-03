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

    response = requests.post("http://localhost:8000/chat", headers=headers, json= data)

    return response.json()