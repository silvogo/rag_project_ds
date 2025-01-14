#from src.chain import load_chain
import os

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.orm.sync import populate

from src.chain import create_chain
from src.retriever import create_vector_store_indexed, populate_vector_store, get_retriever
from src.tokenizer import set_up_tokenizer
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    # Set up tokenizer globally
    set_up_tokenizer()

    # Load and clean the data
    data_file_path = os.getenv("DATA_DIRECTORY")

    # TO DO - Complete load and clean data function
    df = load_and_clean_data(data_file_path)

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # create a vector store and index
    vector_store = create_vector_store_indexed(embedding_model)
    # define token limit
    token_limit = 450
    vector_store = populate_vector_store(vector_store, df, token_limit)

    # Initialize the retriever
    retriever = get_retriever(vector_store)

    chain = create_chain(retriever)

    query = "Can you please tell me the score of the customer A?"
    response = chain.invoke({"input": query})
    print(response['answer'])