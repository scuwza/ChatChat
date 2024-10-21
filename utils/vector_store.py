import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from utils.read_files import process_documents

# 设置本地模型的路径
LOCAL_MODEL_PATH = '/autodl-tmp/bge-large-zh-v1.5'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def initialize_faiss():
    # 加载本地 Hugging Face 模型与 tokenizer
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    # 创建嵌入
    embeddings = HuggingFaceEmbeddings(model=model, tokenizer=tokenizer)

    # 检查并加载 FAISS 数据库
    if not os.path.exists(DB_FAISS_PATH):
        documents = process_documents()
        faiss_db = FAISS.from_documents(documents=documents, embedding=embeddings, persist_directory=DB_FAISS_PATH)
    else:
        faiss_db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    return faiss_db