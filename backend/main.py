import os
import shutil
import uuid

import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse


from backend.src.chain import get_chain
from backend.src.chunking import load_and_chunk
from backend.src.pydantic_models import QueryResponse, QueryInput
from backend.src.retriever import index_document_to_faiss

from dotenv import load_dotenv
import logging


# Start logging
logging.basicConfig(filename="app.log", level=logging.INFO)
# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot Insurance Company")


@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput) -> QueryResponse:
    # Generate session_id if not provided
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}")
    # get the chain created using langchain
    rag_chain = get_chain(query_input.model.value)
    # get answer
    answer = rag_chain.invoke({"input": query_input.question})["answer"]

    logging.info(f"Session ID: {session_id}, Chat Response: {answer}")

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


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

        # Load and chunk the data
        chunks = load_and_chunk(file_path=temp_file_path, chunk_size=1000, chunk_overlap=100)

        # Generate a unique identifier for the file
        file_id = str(uuid.uuid4())

        # Add chunks to FAISS
        success = index_document_to_faiss(chunks, file_id)


        if success:
            return JSONResponse(
                content={"message": f"File {file.filename} has been successfully uploaded and indexed."},
                status_code=200)
        else:
            raise  HTTPException(status_code=500, detail=f"Error processing file: {file.filename}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


# @app.get("/list-documents", response_model=list[DocumentInfo])
# TO DO: FUTURE ENDPOINT TO LIST DOCUMENTS INSERTED
# async def list_documents():
#     """
#     endpoint to list the documents added to the RAG
#     :return:
#     """
#     return

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot for MDS Insurance Company API"}


if __name__ == "__main__":

    load_dotenv()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

