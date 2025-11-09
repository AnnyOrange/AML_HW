import ast
import json
import re
from typing import List, Tuple

import pandas as pd

from retriever import Retriever


# Paths
TRAIN_CSV_PATH = "../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv"
TEST_CSV_PATH = "../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_test.csv"


def parse_conversation_for_qa(conversation_str: str) -> List[Tuple[str, str, str, str]]:
    """Parse conversation into list of (asker, question, answerer, answer) tuples.

    The conversation is assumed to have lines like: "User 1: ..." or "User 2: ...".
    We pair each utterance with the immediate next utterance as QA. This is a
    simple heuristic suitable for this assignment.
    """
    lines = [line.strip() for line in str(conversation_str).splitlines() if line.strip()]
    parsed = []
    user_re = re.compile(r"^(User \d):\s*(.*)$")

    speakers: List[str] = []
    contents: List[str] = []

    for line in lines:
        m = user_re.match(line)
        if not m:
            continue
        speakers.append(m.group(1))
        contents.append(m.group(2))

    qa_pairs: List[Tuple[str, str, str, str]] = []
    for i in range(len(speakers) - 1):
        asker = speakers[i]
        answerer = speakers[i + 1]
        question = contents[i]
        answer = contents[i + 1]
        qa_pairs.append((asker, question, answerer, answer))
    return qa_pairs


def _string_list_from_raw(raw_value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        # Fallback: split lines
        return [line.strip() for line in str(parsed).splitlines() if line.strip()]
    except Exception:
        return [line.strip() for line in str(raw_value).splitlines() if line.strip()]


def find_ground_truth_matches(answer_text: str, ground_truth_personas: List[str]) -> bool:
    """Heuristic: if any keyword from personas appears in the answer, consider it a match.
    This is a very rough heuristic for selecting samples where the answer likely
    references the responder's persona.
    """
    answer_lower = str(answer_text).lower()
    for persona in ground_truth_personas:
        # use simple token overlap heuristic
        tokens = [t for t in persona.lower().split() if len(t) > 3]
        if any(t in answer_lower for t in tokens):
            return True
    return False


def calculate_recall(retrieved: List[str], gt_facts: List[str]) -> float:
    if not gt_facts:
        return 0.0
    r_lower = [r.lower() for r in retrieved]
    hit = 0
    for gt in gt_facts:
        g = str(gt).lower().strip()
        if not g:
            continue
        if any((g in r) or (r in g) for r in r_lower):
            hit = 1
            break
    return float(hit)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--save_trace", type=str, default="")
    parser.add_argument("--engine", type=str, choices=["bm25", "tfidf"], default="bm25", help="选择检索引擎")
    args = parser.parse_args()
    # Prepare test samples
    df_test = pd.read_csv(TEST_CSV_PATH)
    total_hits = 0
    total_samples = 0
    traces = []

    for _, row in df_test.iterrows():
        # 构造本地人设 facts（合并两列）
        u1_personas = _string_list_from_raw(str(row.get("user 1 personas", "")))
        u2_personas = _string_list_from_raw(str(row.get("user 2 personas", "")))
        local_facts = list({p.strip() for p in (u1_personas + u2_personas) if str(p).strip()})
        if not local_facts:
            continue

        # 查询文本优先使用明确的列名，否则回退为从对话抽取QA
        if "user_utterance" in df_test.columns:
            query = str(row.get("user_utterance", "")).strip()
            # 解析 GT：assistant_utterance_grounded_on_persona_facts（多行）
            gt_col = str(row.get("assistant_utterance_grounded_on_persona_facts", ""))
            gt_list = [ln.strip() for ln in gt_col.splitlines() if ln.strip()]
            # 本地检索（按新API：per-dialogue KB）
            retr = Retriever(kb_list=local_facts, engine=args.engine)
            pairs = retr.retrieve(query, top_k=args.top_k)
            retrieved = [t for (t, _) in pairs]
            hit = calculate_recall(retrieved, gt_list)
            total_hits += hit
            total_samples += 1
            traces.append({
                "query": query,
                "retrieved": retrieved,
                "ground_truth": gt_list,
                "hit@k": bool(hit)
            })
        else:
            # 旧数据格式：从对话中抽取 QA，对每个 QA 样本做本地检索 + 旧启发式 GT 过滤
            conv = str(row.get("Best Generated Conversation", ""))
            qa_pairs = parse_conversation_for_qa(conv)
            for asker, question, answerer, answer in qa_pairs:
                # 用启发式选择可能引用人设的样本
                gt_candidates = u1_personas if answerer == "User 1" else u2_personas
                if not gt_candidates:
                    continue
                if not find_ground_truth_matches(answer, gt_candidates):
                    continue
                retr = Retriever(kb_list=local_facts, engine=args.engine)
                pairs = retr.retrieve(question, top_k=args.top_k)
                retrieved = [t for (t, _) in pairs]
                hit = calculate_recall(retrieved, gt_candidates)
                total_hits += hit
                total_samples += 1
                traces.append({
                    "query": question,
                    "retrieved": retrieved,
                    "ground_truth": gt_candidates,
                    "hit@k": bool(hit)
                })

    recall_at_k = (total_hits / total_samples) if total_samples else 0.0
    print(f"评估完成：样本数={total_samples}，Recall@{args.top_k}={recall_at_k:.4f}")

    if args.save_trace:
        with open(args.save_trace, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"检索追踪已保存到 {args.save_trace}")


