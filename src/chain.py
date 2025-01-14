from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def create_chain(vector_store):
    retriever = vector_store.as_retriever()

    system_prompt = """"
            You are an expert analyzing customer satisfaction data.

            Use the given context to answer the question.

            "Context:\n{context}\n\n"
            "Answer as clearly and concisely as possible. If you don't know the answer say you don't know."""

    prompt = PromptTemplate(input_variables=['context', 'input'], template=system_prompt)

    llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
    question_answer_chain = create_stuff_documents_chain(llm, prompt, output_parser=StrOutputParser())
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

    return chain


if __name__ == '__main__':
    print(load_chain())