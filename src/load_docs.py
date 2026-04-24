from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """PDFファイルを読み込み，テキストを返す"""
    reader = PdfReader(file_path)
    texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    return "\n".join(texts)


def load_txt(file_path: str) -> str:
    """txtファイルを読み込み，テキストを返す"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """拡張子に応じてPDFまたはtxtを読み込む"""
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        return load_pdf(file_path)
    elif path.suffix.lower() == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == "__main__":
    file_path = "data/sample.pdf"
    text = load_document(file_path)

    print("===== Loaded Text =====")
    print(text[:2000])