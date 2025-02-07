from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


def load_and_chunk(file_path:str, chunk_size:int, chunk_overlap:int) -> List[Document]:
    # load the file as csv
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    document = loader.load()
    # test the character text splitter
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # split document into chunks
    texts = text_splitter.split_documents(document)

    return texts


