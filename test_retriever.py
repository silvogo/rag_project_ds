import os
from re import search

from dotenv import load_dotenv

from src.chain import vector_store
from src.chunking import load_and_chunk
from src.retriever import index_document_to_faiss, get_vector_store

load_dotenv()

# get file path
csv_file_path = os.getenv('ANSWERS_PATH')
test_query = "Can you please tell me the nps score of the customer XPTO"

def test_retriever():
    try:
        # get vector store
        vector_store = get_vector_store()
        chunks = load_and_chunk(file_path=csv_file_path, chunk_size=1000, chunk_overlap=100)
        print(f"Loaded and chunked {len(chunks)} documents")
        for idx, chunk in enumerate(chunks[:5]):
            print(f"Chunk {idx + 1} Metadata: {chunk.metadata}")

        # index chunks into FAISS
        file_id = 'test_file'
        success = index_document_to_faiss(chunks, file_id)
        print(f"FAISS index size: {vector_store.index.ntotal}")
        print(f"Is FAISS index writable? {vector_store.index.is_trained}")

        if not success:
            print("Failed to index documents into FAISS")
            return

        print("Successfully indexed documents into FAISS")


        # Test the retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke(test_query)

        print("\nRetriever results")
        print(results)

        for i, doc in enumerate(results):
            print(f"Result {i + 1 }")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")


    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == '__main__':
    test_retriever()