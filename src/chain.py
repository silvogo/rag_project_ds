from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from src.retriever import create_vector_store_indexed

# Load environment variables
# load_dotenv()


embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vector_store = create_vector_store_indexed(embedding_function)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# Define the prompt for question answering
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert analyzing customer satisfaction data.
            Use the given context to answer the question.
            <context>
            {context}
            </context>  
            If you don't know the answer, say you don't know.
            """,
        ),
        ("human", "{input}"),
    ]
)


def get_chain(model="gpt-4o-mini"):

    # define the llm
    llm = ChatOpenAI(temperature=0, model_name=model)
    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=qa_prompt, output_parser=StrOutputParser()
    )

    chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )

    return chain
