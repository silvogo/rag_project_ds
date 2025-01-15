import os

from src.chain import create_chain
from src.data_processing import load_and_clean_data
from src.retriever import create_vector_store_indexed, populate_vector_store, get_retriever

from dotenv import load_dotenv
from transformers import AutoTokenizer
from llama_index.core import set_global_tokenizer
from langchain_huggingface import HuggingFaceEmbeddings


if __name__ == '__main__':
    load_dotenv()
    # Set up tokenizer globally
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    set_global_tokenizer(tokenizer)
    print("Tokenizer is set!")

    # Load and clean the data
    data_file_path = os.getenv("DATA_DIRECTORY")
    print("LOADING AND CLEANING DATA")
    # Load and clean data
    df = load_and_clean_data()

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    print("CREATING VECTOR STORE")
    # create a vector store and index
    vector_store = create_vector_store_indexed(embedding_model)
    print(f"Type after creation: {type(vector_store)}")
    # define token limit
    token_limit = 450
    # populate vector store
    print("POPULATING VECTOR STORE")
    vector_store = populate_vector_store(vector_store, df, tokenizer, token_limit)

    # Initialize the retriever
    retriever = get_retriever(vector_store)

    chain = create_chain(retriever)

    query = "Can you please tell me one improvement suggestion provided by a customer?"
    response = chain.invoke({"input": query})
    print(response['answer'])