import os
import shutil
import uuid

import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse


from backend.src.chain import get_chain
from backend.src.chunking import load_and_chunk
from backend.src.database_utils import get_all_documents, insert_document_record, delete_document_record, \
    get_chat_history, insert_application_logs
from backend.src.pydantic_models import QueryResponse, QueryInput, DeleteFileRequest, DocumentInfo
from backend.src.retriever import index_document_to_faiss, delete_doc_from_faiss

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

    # get chat history
    chat_history = get_chat_history(session_id)

    # get the chain created using langchain
    rag_chain = get_chain(query_input.model.value)
    # get answer
    answer = rag_chain.invoke({"input": query_input.question,
                               "chat_history": chat_history})["answer"]
    # insert application logs into embedded database
    insert_application_logs(session_id=session_id, user_query=query_input.question,
                            model_response=answer, model= query_input.model.value )

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

        # insert document into database
        file_id = insert_document_record(file.filename)
        # Load and chunk the data
        chunks = load_and_chunk(file_path=temp_file_path, chunk_size=1000, chunk_overlap=200)
        # Add chunks to FAISS
        success = index_document_to_faiss(chunks, file_id)

        if success:
            return JSONResponse(
                content={"message": f"File {file.filename} has been successfully uploaded and indexed."},
                status_code=200)
        else:
            delete_document_record(file_id)
            raise  HTTPException(status_code=500, detail=f"Error processing file: {file.filename}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


@app.get("/list-documents", response_model=list[DocumentInfo])
async def list_documents():
    """
    endpoint to list the documents added to the RAG
    :return:
    """
    return get_all_documents()


@app.post("/delete-doc")
async def delete_document(request:DeleteFileRequest):
    # Get file_id
    file_id = request.file_id

    # Delete from FAISS
    delete_success_faiss = delete_doc_from_faiss(file_id=file_id)
    if not delete_success_faiss:
        # Return error if FAISS deletion fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file with id {file_id} from FAISS."
        )

    # Delete from Database
    deleted_from_db = delete_document_record(request.file_id)
    if not deleted_from_db:
        # Return error if database deletion fails after FAISS deletion
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file with id {file_id} from the database after deleting from FAISS."
        )

    # Success Response
    return JSONResponse(
        content={
            "message": f"File with id {file_id} has been successfully deleted from FAISS and database."
        },
        status_code=200
    )

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot for MDS Insurance Company API"}


if __name__ == "__main__":

    load_dotenv()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

