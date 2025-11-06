import argparse


def run_lightrag(source_path: str):
    import os
    import logging

    from lightrag import LightRAG, QueryParam
    from lightrag.llm import zhipu_complete, zhipu_embedding
    from lightrag.utils import EmbeddingFunc

    WORKING_DIR = "./working_dir"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if api_key is None:
        raise Exception("Please set ZHIPUAI_API_KEY in your environment")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=zhipu_complete,
        llm_model_name="glm-4-flash",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=2048,
            max_token_size=8192,
            func=lambda texts: zhipu_embedding(texts),
        ),
    )

    def _ingest_path(path: str):
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    full = os.path.join(root, name)
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as f:
                            rag.insert(f.read())
                    except Exception:
                        continue
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                rag.insert(f.read())

    _ingest_path(source_path)

    print(rag.query("What is FRP?", param=QueryParam(mode="naive")))
    print(rag.query("What is FRP?", param=QueryParam(mode="local")))
    print(rag.query("What is FRP?", param=QueryParam(mode="global")))
    print(rag.query("What is FRP?", param=QueryParam(mode="hybrid")))


def run_bm25(top_k: int, verbose: bool, kb_path: str, model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from retriever import BM25Retriever
    import pandas as pd

    MODEL_PATH = model_path or "./my_finetuned_model"
    KNOWLEDGE_BASE_PATH = kb_path or "../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv"

    # 从传入CSV随机抽取一行，构造该行的“本地人设事实集合”与查询
    try:
        df = pd.read_csv(KNOWLEDGE_BASE_PATH)
        sample = df.sample(n=1, random_state=None).iloc[0]
        u1 = sample.get("user 1 personas", "")
        u2 = sample.get("user 2 personas", "")
        local_facts = []
        local_facts.extend(str(u1).splitlines())
        local_facts.extend(str(u2).splitlines())
        local_facts = [s.strip() for s in local_facts if str(s).strip()]
        if not local_facts:
            local_facts = ["I like hiking.", "I am a teacher."]
        # 选择查询：优先 user_utterance/question，否则给定一个简短问句
        user_query = str(sample.get("user_utterance", "")).strip() or str(sample.get("question", "")).strip()
        if not user_query:
            user_query = "What do you do for fun?"
        # 目标检索（若数据提供）与目标输出
        gt_facts_raw = str(sample.get("assistant_utterance_grounded_on_persona_facts", "")).strip()
        target_facts = [ln.strip() for ln in gt_facts_raw.splitlines() if ln.strip()] if gt_facts_raw else []
        target_answer = str(sample.get("assistant_utterance", "")).strip() or str(sample.get("answer", "")).strip()
    except Exception:
        local_facts = ["I like hiking.", "I am a teacher."]
        user_query = "What do you do for fun?"
        target_facts = []
        target_answer = ""
    retriever = BM25Retriever(local_persona_facts=local_facts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)

    results = retriever.retrieve_with_scores(user_query, top_k=top_k)
    retrieved_personas = [p for p, _ in results]

    if verbose:
        print("--- 检索TopK（含BM25分数）---")
        for p, s in results:
            print(f"{s:.4f} :: {p}")

    numbered = [f"Persona [{i+1}]: {p}" for i, p in enumerate(retrieved_personas)]
    prompt_context = "[CONTEXT]\n" + "\n".join(numbered)
    final_prompt = f"{prompt_context}\n\n[QUESTION]\n{user_query}\n\n[ANSWER]\n"

    response = generator(final_prompt)
    generated = response[0]["generated_text"].replace(final_prompt, "")

    # 按需求打印：1) 问题 2) 目标检索 3) 目标输出 4) 实际检索 5) 实际输出
    print("\n===== 1) 问题 =====")
    print(user_query)

    print("\n===== 2) 目标检索（Ground Truth） =====")
    if target_facts:
        for i, f in enumerate(target_facts, 1):
            print(f"GT[{i}]: {f}")
    else:
        print("(无 ground-truth 列，或当前 CSV 未提供)")

    print("\n===== 3) 目标输出（Ground Truth Answer） =====")
    print(target_answer if target_answer else "(无 ground-truth 列，或当前 CSV 未提供)")

    print("\n===== 4) 实际检索（BM25 本地 TopK） =====")
    for p, s in results:
        print(f"{s:.4f} :: {p}")

    print("\n===== 5) 实际输出（模型生成） =====")
    print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["lightrag", "bm25"], default="lightrag")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--lightrag_path", type=str, default="./book.txt")
    parser.add_argument("--kb_path", type=str, default="../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv")
    parser.add_argument("--model_path", type=str, default="./my_finetuned_model")
    args = parser.parse_args()

    if args.engine == "lightrag":
        run_lightrag(source_path=args.lightrag_path)
    else:
        run_bm25(top_k=args.top_k, verbose=bool(args.verbose), kb_path=args.kb_path, model_path=args.model_path)
