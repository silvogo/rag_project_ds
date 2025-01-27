import os
import faiss
import pickle
from dotenv import load_dotenv

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# start vector store as none
_vector_store = None
# Define faiss index path
FAISS_INDEX_PATH = 'faiss_index'
DOCSTORE_PATH ='docstore'
INDEX_TO_DOCSTORE_ID = 'index_to_docstore_id'


def save_vector_store(vector_store, index_path=FAISS_INDEX_PATH, id_map_path=INDEX_TO_DOCSTORE_ID, docstore_path=DOCSTORE_PATH) :
    """
    Saves the FAISS index, document store, and index-to-docstore ID mapping to disk.

    :param vector_store: The FAISS vector store object to save.
    :param index_path: The file path to save the FAISS index.
    :param id_map_path: The file path to save the index-to-docstore ID mapping.
    :param docstore_path: The file path to save the document store.

    :return: None
    """
    # save index to the current dir
    faiss.write_index(vector_store.index, index_path)

    with open(docstore_path, "wb") as f:
        pickle.dump(vector_store.docstore, f)

    with open(id_map_path, "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)


def load_vector_store (index_path, docstore_path, id_map_path):
    """
   Loads an existing FAISS vector store, including the index, document store,
    and index-to-docstore ID mapping.
    :param index_path: The file path to the FAISS index.
    :param docstore_path: The file path to the document store.
    :param id_map_path: The file path to the index-to-docstore ID mapping.
    :return: A tuple containing the FAISS index, document store, and index-to-docstore ID mapping.
    """
    # reading index path
    index = faiss.read_index(index_path)
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)
    with open(id_map_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    return index, docstore, index_to_docstore_id



# Function to initialize vector store indexed
def initialize_vector_store_indexed(embeddings, index_path = FAISS_INDEX_PATH, id_map_path=INDEX_TO_DOCSTORE_ID,
                                    docstore_path=DOCSTORE_PATH):
    """
    Initializes a FAISS vector store, either by loading an existing index or creating a new one.
    :param embeddings: The embedding function to use for generating embeddings.
    :param index_path: The file path to the FAISS index. Defaults to FAISS_INDEX_PATH.
    :param id_map_path: The file path to the index-to-docstore ID mapping. Defaults to INDEX_TO_DOCSTORE_ID.
    :param docstore_path: The file path to the document store. Defaults to DOCSTORE_PATH.
    :return: vector_store: An initialized FAISS vector store object
    """
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        # loading faiss index, docstore and index_to_docstore_id
        index, docstore, index_to_docstore_id = load_vector_store(index_path, docstore_path, id_map_path)
        print("FAISS index loaded successfully")
    else:
        print("Creating a new FAISS index...")
        # Generate a sample embedding to determine dimensionality
        sample_embedding = embeddings.embed_query("test")
        dimension = len(sample_embedding)

        # Create a FAISS index for L2 distance
        index = faiss.IndexFlatL2(dimension)
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}


    # Initialize the FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vector_store



# Function to index documents to faiss
def index_document_to_faiss (chunks, file_id):
    """
    Indexes a set of document chunks into the global FAISS vector store and persists it to disk.

    :param chunks: A list of document chunks to index.
    :param file_id: A unique identifier for the file being indexed.
    :return: bool: True if indexing was successful, False otherwise.
    """
    global _vector_store
    print("Entering index document to faiss function")
    try:
        for chunk in chunks:
            # assign id as a key to metadata attribute (which is a dict)
            chunk.metadata['file_id'] = file_id

        # add documents (as chunks) to vectorstore
        _vector_store.add_documents(chunks)

        # Persist the FAISS index to disk
        save_vector_store(_vector_store)
        print("FAISS index, docstore and index_to_docstore_id saved to disk.")
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def get_vector_store():
    """
    Retrieves the global FAISS vector store. Initializes it if it does not already exist.

    :return: _vector_store: The global FAISS vector store object
    """
    global _vector_store
    if _vector_store is None:
        embeddings =  OpenAIEmbeddings()
        _vector_store = initialize_vector_store_indexed(embeddings=embeddings)
    return _vector_store

# load environment variables
load_dotenv()

