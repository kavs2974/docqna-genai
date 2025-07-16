from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
import tempfile

def load_document(uploaded_file):
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    else:
        loader = Docx2txtLoader(tmp_file_path)

    data = loader.load()
    os.remove(tmp_file_path)
    return data

def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks

def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
