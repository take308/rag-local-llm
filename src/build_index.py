from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from load_docs import load_document
from split_docs import split_text


DATA_PATH = "data/sample.pdf"
VECTORSTORE_DIR = "vectorstore/faiss_index"


def build_index():
    text = load_document(DATA_PATH)
    chunks = split_text(text)

    print(f"チャンク数: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    Path(VECTORSTORE_DIR).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"FAISS index saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_index()