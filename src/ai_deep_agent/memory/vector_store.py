"""
RAG Vector Store — Documents store aur retrieve karne ke liye.
"""
# from __future__ import annotations
# import os
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_core.documents import Document

# EMBEDDINGS = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# CHROMA_PATH = "./chroma_db"

# def get_vector_store() -> Chroma:
#     return Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=EMBEDDINGS,
#     )

# def ingest_pdf(file_path: str) -> int:
#     loader = PyPDFLoader(file_path)
#     docs   = loader.load()
#     return _chunk_and_store(docs)

# def ingest_text(text: str, source_name: str = "user_input") -> int:
#     docs = [Document(page_content=text, metadata={"source": source_name})]
#     return _chunk_and_store(docs)

# def ingest_url(url: str) -> int:
#     loader = WebBaseLoader(url)
#     docs   = loader.load()
#     return _chunk_and_store(docs)

# def retrieve_context(query: str, k: int = 4) -> str:
#     store = get_vector_store()
#     docs  = store.similarity_search(query, k=k)
#     if not docs:
#         return ""
#     return "\n\n---\n\n".join(
#         f"[Source: {d.metadata.get('source', 'document')}]\n{d.page_content}"
#         for d in docs
#     )

# def clear_vector_store() -> None:
#     import shutil
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

# def _chunk_and_store(docs) -> int:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#     )
#     chunks = splitter.split_documents(docs)
#     store  = get_vector_store()
#     store.add_documents(chunks)
#     return len(chunks)





# modified
from __future__ import annotations
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

CHROMA_PATH = "/tmp/chroma_db"
# ------------------------------------------------------------------ #
#  Lazy embedding loader                                              #
#  HuggingFace model sirf tab load hoga jab pehli baar zarurat ho   #
# ------------------------------------------------------------------ #
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openai_api_base="https://openrouter.ai/api/v1",
        )
    return _embeddings

def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=_get_embeddings(),
    )

def ingest_pdf(file_path: str) -> int:
    loader = PyPDFLoader(file_path)
    docs   = loader.load()
    return _chunk_and_store(docs)

def ingest_text(text: str, source_name: str = "user_input") -> int:
    docs = [Document(page_content=text, metadata={"source": source_name})]
    return _chunk_and_store(docs)

def ingest_url(url: str) -> int:
    loader = WebBaseLoader(url)
    docs   = loader.load()
    return _chunk_and_store(docs)

def retrieve_context(query: str, k: int = 4) -> str:
    store = get_vector_store()
    docs  = store.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'document')}]\n{d.page_content}"
        for d in docs
    )

def clear_vector_store() -> None:
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def _chunk_and_store(docs) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    store  = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)