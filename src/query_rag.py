import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


VECTORSTORE_DIR = "vectorstore/faiss_index"

# 軽量モデルから開始
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def search_documents(query: str, k: int = 3):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def build_prompt(query: str, docs):
    context = "\n\n".join(
        [f"[文書{i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
以下の参考文書だけを根拠にして、日本語で質問に答えてください。
参考文書に答えがない場合は、「文書中には明確に書かれていません」と答えてください。

# 参考文書
{context}

# 質問
{query}

# 回答
"""
    return prompt


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


def generate_answer(prompt: str, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": "あなたは文書に基づいて正確に回答する日本語アシスタントです。",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


if __name__ == "__main__":
    query = input("質問を入力してください: ")

    docs = search_documents(query, k=3)
    prompt = build_prompt(query, docs)

    tokenizer, model = load_llm()
    answer = generate_answer(prompt, tokenizer, model)

    print("\n===== 回答 =====")
    print(answer)

    print("\n===== 参照チャンク =====")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:500])