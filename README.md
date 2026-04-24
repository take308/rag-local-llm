# RAG Local LLM

ローカル環境で完結して動作する RAG（Retrieval-Augmented Generation）システムです。  
外部APIを使用せず、PDFやテキスト文書を元にした高精度な質問応答をローカルPC上で行うことができます。

## 🚀　概要

本プロジェクトでは、手元の文書を適切なサイズに分割してベクトル化し、質問に対して関連性の高い部分を高速に検索（ベクトル検索）した上で、ローカルLLMを用いて回答を生成します。

### 処理の流れ

1.  **文書入力**: PDF または `.txt` ファイルの読み込み
2.  **テキスト抽出**: 文書からのテキストデータの取り出し
3.  **チャンク分割**: 長い文章を検索しやすいサイズに分割
4.  **Embedding**: `bge-m3` モデルを使用してテキストをベクトル化
5.  **ベクトルDB保存**: `FAISS` を使用してベクトルをインデックス化
6.  **質問入力**: ユーザーからの問いをベクトル化して検索
7.  **類似検索**: 関連するチャンク（文書断片）を抽出
8.  **回答生成**: `Qwen` モデルが検索結果を元に回答を作成

## 使用技術

- Language: Python
- Vector DB: [FAISS](https://github.com/facebookresearch/faiss)
- Embedding: [sentence-transformers](https://www.sbert.net/) (BAAI/bge-m3)
- LLM: [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) (Qwen/Qwen2.5)
- Framework: [LangChain](https://python.langchain.com/)

## ⚙️ 環境構築

```bash
uv venv
. .venv/bin/activate