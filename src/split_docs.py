from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_docs import load_document


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """長い文章をチャンクに分割する"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    file_path = "data/sample.pdf"

    text = load_document(file_path)
    chunks = split_text(text)

    print(f"チャンク数: {len(chunks)}")
    for i in range(min(5, len(chunks))):  # 最初の5つのチャンクを表示
        print("=====" + str(i + 1) + "つ目のチャンク =====")
        print(chunks[i])