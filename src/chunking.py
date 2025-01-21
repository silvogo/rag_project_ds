import hashlib
from typing import List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import set_global_tokenizer, Document as Llama_Index_Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



def chunk_customer_info (customer_info_column: pd.Series, chunk_size: int = 1000, chunk_overlap: int = 100) -> pd.Series:
    """
        Splits the customer_info column into chunks using Langchain's RecursiveCharacterTextSplitter.
        Returns a Pandas Series where each row is a list of chunks corresponding to the original row.
        This function will be tested as new method for chunking
    """
    # define splitter for chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = customer_info_column.apply(lambda customer_info: splitter.split_text(customer_info))

    return pd.Series(chunks)

# Function to split text by sentences using llama index
def split_long_text(text, chunksize=1000, overlap=100):
    """
    Splits a long text into smaller chunks using a sentence splitter.
    """
    # Set up a sentence splitter
    sentence_splitter = SentenceSplitter(chunk_size=chunksize, chunk_overlap=overlap)
    document = Llama_Index_Document(text=text)
    nodes = sentence_splitter.get_nodes_from_documents([document])

    # Return the text of each chunk
    return [node.text for node in nodes]



# TO DEPRECATE
def combine_customer_info(
    customer_name,
    scores,
    nps_score,
    nps_type,
    open_responses,
    tokenizer,
    token_limit=450,
):
    """
    Combines customer data and splits it into chunks if it exceeds the token limit.
    """
    customer_info = []

    # Create the base text with customer name, scores, and NPS score
    base_text = (
        f"Customer: {customer_name}\nScores:\n{scores}\n{nps_score}\n{nps_type}\n"
    )

    # Tokenize the base text
    base_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))

    # Tokenize the total text including open responses
    total_text = base_text + open_responses
    total_tokens = len(tokenizer.encode(total_text, add_special_tokens=False))

    if total_tokens > token_limit:
        # Remaining tokens for the first chunk after adding the base text
        remaining_tokens = token_limit - base_tokens

        # Safety margin
        remaining_tokens = remaining_tokens - 30

        # Split open responses to fit within the remaining tokens
        open_response_chunks = split_long_text(open_responses, remaining_tokens)

        # Combine the first chunk of open responses with the base text
        first_chunk = base_text + open_response_chunks[0]
        customer_info.append(first_chunk)

        # Add the remaining open response chunks as separate documents
        for chunk in open_response_chunks[1:]:
            customer_info.append(chunk)
    else:
        # If everything fits within the token limit, add as a single document
        customer_info.append(total_text)

    return customer_info


def generate_hash(customer_name, chunk_index):
    base_string = f"{customer_name}_chunk_{chunk_index}"
    return hashlib.md5(base_string.encode()).hexdigest
