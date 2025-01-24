import os
import faiss
from dotenv import load_dotenv

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# start vector store as none
_vector_store = None
# Define faiss index path
FAISS_INDEX_PATH = 'faiss_index'


# Function to initialize vector store indexed
def initialize_vector_store_indexed(embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("FAISS index loaded successfully")
    else:
        print("Creating a new FAISS index...")
        # Generate a sample embedding to determine dimensionality
        sample_embedding = embeddings.embed_query("test")
        dimension = len(sample_embedding)

        # Create a FAISS index for L2 distance
        index = faiss.IndexFlatL2(dimension)

    # Initialize the FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store



# Function to index documents to faiss
def index_document_to_faiss (chunks, file_id):
    global _vector_store
    try:
        for chunk in chunks:
            # assign id as a key to metadata attribute (which is a dict)
            chunk.metadata['file_id'] = file_id

        # add documents (as chunks) to vectorstore
        _vector_store.add_documents(chunks)

        # Persist the FAISS index to disk
        faiss.write_index(_vector_store.index, FAISS_INDEX_PATH)
        print("FAISS index saved to disk.")
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings =  OpenAIEmbeddings()
        _vector_store = initialize_vector_store_indexed(embeddings=embeddings)
    return _vector_store

load_dotenv()


# define the initial load of vector store
# TO DEPRECATE (?)
# def populate_vector_store(vector_store, source_documents_df, tokenizer, token_limit):
#     # Process customer data
#     for _, row in source_documents_df.iterrows():
#         customer_name = row["customer"]
#         scores = row["scores"]
#         nps_score = row["nps_score"]
#         nps_type = row["nps_type"]
#         open_responses = row["open_responses"]
#
#         # Split customer data into manageable chunks
#         chunks_customer = combine_customer_info(
#             customer_name,
#             scores,
#             nps_score,
#             nps_type,
#             open_responses,
#             tokenizer=tokenizer,
#             token_limit=token_limit,
#         )
#
#         # Add metadata and store chunks
#         documents = [
#             LangchainDocument(
#                 page_content=chunk, metadata={"customer": customer_name, "part": i + 1}
#             )
#             for i, chunk in enumerate(chunks_customer)
#         ]
#         uuids = [generate_hash(customer_name, i + 1) for i in range(len(documents))]
#         vector_store.add_documents(documents, ids=uuids)
#
#     return vector_store

