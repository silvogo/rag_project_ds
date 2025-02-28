from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


def load_and_chunk(file_path:str, chunk_size:int, chunk_overlap:int) -> List[Document]:
    # TO DO: Improving chunking approach
    # load the file as csv. Each row will be treated as a single Document object
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    # Document.page_content will contain the row's values as a single text string
    document = loader.load()
    # test the character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   length_function=len, separators= ["\n\n", "\n", " ", ""])
    # split document into chunks
    texts = text_splitter.split_documents(document)

    return texts


