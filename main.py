import os
import shutil
import pandas as pd

import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse

from src.chain import get_chain
from src.chunking import chunk_customer_info
from src.data_processing import load_and_clean_data
from src.pydantic_models import QueryResponse, QueryInput
from src.retriever import (
    create_vector_store_indexed,
    populate_vector_store,
)

from dotenv import load_dotenv
from transformers import AutoTokenizer
from llama_index.core import set_global_tokenizer
from langchain_huggingface import HuggingFaceEmbeddings
import logging


# Start logging
logging.basicConfig(filename="app.log", level=logging.INFO)
# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot")


@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput) -> QueryResponse:
    session_id = query_input.session_id
    # get the chain created using langchain
    rag_chain = get_chain(query_input.model.value)
    answer = rag_chain.invoke({"input": query_input.question})["answer"]

    logging.info(f"Session ID: {session_id}, Chat Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.mode)

@app.post("/upload-doc")
async def upload_documents(file: UploadFile=File(...)):
    """
    Endpoint to upload a document (e.g., CSV).
    Processes and adds the documents to FAISS
    :param file:
    :return:
    """
    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and clean the data
        df = pd.read_csv(temp_file_path)
        # Clean the data
        cleaned_df = load_and_clean_data(df)

        # Chunk the data for FAISS - TO REFINE
        cleaned_df['chunks'] = chunk_customer_info(cleaned_df['customer_info'])
        # Add chunks to FAISS - TO REFINE
        # add_to_faiss(chunks, retriever)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"message": f"File {file.filename} has been successfully uploaded and indexed."}, status_code=200)

    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint for health check
    :return:
    """
    return {"message": "RAG Chatbot is up and running!"}


if __name__ == "__main__":

    load_dotenv()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

