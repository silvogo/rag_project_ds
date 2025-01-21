import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from src.chunking import combine_customer_info, generate_hash


def create_vector_store_indexed(embedding_function):
    # get an embeddings model

    # Generate a sample embedding to determine dimensionality
    sample_embedding = embedding_function.embed_query("test")
    dimension = len(sample_embedding)

    # Create a FAISS index for L2 distance
    index = faiss.IndexFlatL2(dimension)

    # Initialize an empty FAISS vector store
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store


# define the initial load of vector store
# THIS FUNCTION NEEDS REFINEMENT
def populate_vector_store(vector_store, source_documents_df, tokenizer, token_limit):
    # Process customer data
    for _, row in source_documents_df.iterrows():
        customer_name = row["customer"]
        scores = row["scores"]
        nps_score = row["nps_score"]
        nps_type = row["nps_type"]
        open_responses = row["open_responses"]

        # Split customer data into manageable chunks
        chunks_customer = combine_customer_info(
            customer_name,
            scores,
            nps_score,
            nps_type,
            open_responses,
            tokenizer=tokenizer,
            token_limit=token_limit,
        )

        # Add metadata and store chunks
        documents = [
            LangchainDocument(
                page_content=chunk, metadata={"customer": customer_name, "part": i + 1}
            )
            for i, chunk in enumerate(chunks_customer)
        ]
        uuids = [generate_hash(customer_name, i + 1) for i in range(len(documents))]
        vector_store.add_documents(documents, ids=uuids)

    return vector_store
