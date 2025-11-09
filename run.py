import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from retriever import Retriever
try:
    from peft import PeftModel  # 可选
except Exception:
    PeftModel = None


def build_prompt(persona_context: list, history: str, question: str, answer_with_tags: str = None):
    """
    根据 HW1数据说明.pdf 构建 prompt 和 completion (与 finetune.py 保持一致)
    """
    persona_str = "\n".join(persona_context)
    prompt = f"Your Personas:\n{persona_str}\n\n"
    prompt += f"Conversation History:\n{history}\n\n"
    prompt += f"User 2's response:\n{question}\n\n"  # User 2 是提问者
    prompt += "User 1's response:\n"
    if answer_with_tags:
        completion = f"{prompt}{answer_with_tags}"
        return completion
    return prompt


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model checkpoint")
    parser.add_argument("--kb_path", type=str, required=True, help="Path to the test/validation CSV file")
    parser.add_argument("--engine", type=str, default="bm25", help="Retrieval engine (bm25, tfidf)")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # 可视化/预览与追踪
    parser.add_argument("--print_prompt", action="store_true", help="打印用于生成的完整 Prompt")
    parser.add_argument("--save_trace", type=str, default=None, help="将每条RAG过程以JSONL保存到该路径")
    parser.add_argument("--preview_n_dialogues", type=int, default=0, help="仅可视化前N场对话后退出（0表示处理全部）")
    parser.add_argument("--max_pairs_per_dialogue", type=int, default=2, help="每场对话最多展示的QA样例数")
    args = parser.parse_args()
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded.")
    
    try:
        data = pd.read_csv(args.kb_path).fillna("")
    except Exception as e:
        print(f"Error loading {args.kb_path}: {e}")
        return
        
    print(f"Processing {args.kb_path}...")
    
    printed_dialogues = 0
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Testing Dialogues"):
        kb_personas_str = str(row.get("user 1 personas", ""))
        kb_list = [p.strip() for p in kb_personas_str.split('\n') if p.strip()]
        if len(kb_list) < args.top_k:
            continue
        try:
            retriever = Retriever(kb_list=kb_list, engine=args.engine)
        except Exception as e:
            print(f"Error initializing retriever for row {index}: {e}")
            continue
        conversation_lines = str(row.get("Best Generated Conversation", "")).split('\n')
        conversation_history = []
        print(f"\n===== [Dialogue {index+1}] =====")
        if args.verbose:
            print(f"{Colors.GREEN}===== 0) 知识库 (KB) (User 1 Personas) ====={Colors.ENDC}")
            for i, p in enumerate(kb_list):
                print(f"[{i+1}] {p}")
            print("-" * 20)
        pairs_printed = 0
        for i in range(len(conversation_lines) - 1):
            line = conversation_lines[i].strip()
            next_line = conversation_lines[i+1].strip()
            if line.startswith("User 2:") and next_line.startswith("User 1:"):
                question = line.replace("User 2:", "").strip()
                ground_truth_answer = next_line.replace("User 1:", "").strip()
                history_context = "\n".join(conversation_history)
                if not question:
                    continue
                retrieved_results = retriever.retrieve(question, top_k=args.top_k)
                retrieved_personas_text = [res[0] for res in retrieved_results]
                prompt = build_prompt(
                    persona_context=retrieved_personas_text,
                    history=history_context,
                    question=question
                )
                if args.print_prompt:
                    print(f"{Colors.YELLOW}===== PROMPT (用于生成) ====={Colors.ENDC}")
                    print(prompt if len(prompt) < 2000 else (prompt[:2000] + "... [truncated]"))
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                print(f"\n{Colors.YELLOW}===== 1) 问题 (User 2) ====={Colors.ENDC}")
                print(question)
                if args.verbose:
                    print(f"{Colors.GREEN}===== 2) 目标检索 (Ground Truth KB) ====={Colors.ENDC}")
                    print("(注: 目标KB为全部 User 1 Personas，见上方 '0) 知识库')")
                    print(f"{Colors.GREEN}===== 3) 目标输出 (Ground Truth Answer) ====={Colors.ENDC}")
                    print(ground_truth_answer)
                    print(f"{Colors.YELLOW}===== 4) 实际检索 (Top-{args.top_k} {args.engine}) ====={Colors.ENDC}")
                    for text, idx in retrieved_results:
                        print(f"[{idx}] {text}")
                print(f"{Colors.RED}===== 5) 实际输出 (模型生成) ====={Colors.ENDC}")
                print(generated_text)
                pairs_printed += 1
                if args.max_pairs_per_dialogue and pairs_printed >= args.max_pairs_per_dialogue:
                    break
                # 可选：保存追踪
                if args.save_trace:
                    try:
                        trace_rec = {
                            "dialogue_index": int(index),
                            "question": question,
                            "ground_truth_answer": ground_truth_answer,
                            "history": history_context,
                            "kb_list": kb_list,
                            "retrieved": [{"idx": int(idx), "text": text} for text, idx in retrieved_results],
                            "prompt": prompt,
                            "generated": generated_text,
                            "engine": args.engine,
                            "top_k": int(args.top_k),
                        }
                        # 逐行写入 JSONL
                        with open(args.save_trace, "a", encoding="utf-8") as f:
                            f.write(json.dumps(trace_rec, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[Trace] Failed to append to {args.save_trace}: {e}")
                conversation_history.append(line)
                conversation_history.append(next_line)
            elif line.startswith("User 1:") or line.startswith("User 2:"):
                conversation_history.append(line)
        printed_dialogues += 1
        if args.preview_n_dialogues and printed_dialogues >= args.preview_n_dialogues:
            print(f"\n[Preview] 已预览前 {printed_dialogues} 场对话，提前结束。")
            break


if __name__ == "__main__":
    main()
