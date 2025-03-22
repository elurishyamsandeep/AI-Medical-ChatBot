from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#phase 1 the is first part 

DATA_PATH = r"D:\gopractice\New folder\chatmodel\data"

def load_pdf_files(data):
    loader= DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("length is ",len(documents))
# create chunks 

def cchunks(extrated_data):
    text_splitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extrated_data)
    return text_chunks

text_chunks=cchunks(extrated_data=documents)
print('lent of chunks is ',len(text_chunks))
# create vector db

def get_emb_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_emb_model()
DB_FAISS = r"D:\gopractice\New folder\chatmodel\vectorstore\db_fiass"
db= FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS)


