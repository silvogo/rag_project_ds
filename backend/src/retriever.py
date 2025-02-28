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
    # In the future explore other options like IVFFlat (Inverted File Index Flat)
    # or IVFPQ (Inverted File with Product Quantization)
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

def delete_doc_from_faiss(file_id: int):
    global _vector_store

    try:
        if _vector_store is None:
            _vector_store = get_vector_store()

        print("Deleting from FAIIS")

        # declare documents to delete
        doc_ids_to_delete = []

        # Iterate over stored document IDs
        for doc_id in _vector_store.docstore._dict:  # Access stored document IDs
            doc = _vector_store.docstore.search(doc_id)  # Fetch the document

            if doc and "file_id" in doc.metadata and doc.metadata["file_id"] == file_id:
                doc_ids_to_delete.append(doc_id)

        if not doc_ids_to_delete:
            print(f"No documents found for file_id {file_id}")
            return False


        # delete from faiss index and docstore
        _vector_store.delete(doc_ids_to_delete)
        # Persist changes to disk
        save_vector_store(_vector_store)

        return True

    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from FAISS: {str(e)}")
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

def get_retriever(vector_store, search_type=None, k=5 , lambda_mult=0.5, fetch_k=None, score_threshold=None):
    """
        Returns a retriever based on the search type.

        Args:
            vector_store: FAISS vector store.
            search_type (str): The type of retrieval. Options: 'mmr', 'similarity_score_threshold', or None (default L2).
            k (int): Number of results to return.
            lambda_mult (float): Balances relevance & diversity for MMR. Default 0.5.
            fetch_k (int): Number of documents to fetch initially in MMR.
            score_threshold (float): Minimum similarity score for similarity_score_threshold search.

        Returns:
            retriever: Configured retriever.
    """

    if search_type is None:
        # Default FAISS similarity search - L2 distance

        faiss.normalize_L2(vector_store.index) # ensure cosine similarity if needed
        retriever = vector_store.as_retriever(
            search_kwargs ={"k": k}
        )

    # using maximum marginal relevance algorith
    elif search_type == 'mmr':
        if fetch_k is None:
            fetch_k = k * 10 # Default: Fetch more documents for reranking

        retriever = vector_store.as_retriever(
            search_type= search_type,
            search_kwargs={"k": k, 'lambda_mult': lambda_mult, "fetch_k": fetch_k}
        )

    elif search_type == 'similarity_score_threshold':
        if score_threshold is None:
            raise ValueError("score_threshold must be specified for similarity_score_threshold")
        retriever = vector_store.as_retriever(
            search_type = search_type,
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}. Use 'mmr', 'similarity_score_threshold' or do not specify ")


    return retriever

load_dotenv()


