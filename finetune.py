import ast
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
    BitsAndBytesConfig,
)
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import json
try:
    import bitsandbytes as bnb  # 可选
except Exception:
    bnb = None
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # 可选
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
try:
    from trl import SFTTrainer  # 可选
except Exception:
    SFTTrainer = None
from trl import SFTTrainer
import os
from dataclasses import dataclass
from retriever import BM25Retriever

# 预览缓存（用于可视化检查，最多缓存100条）
FINETUNE_PREVIEW: List[Dict] = []

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
TRAIN_CSV_PATH = "../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv"
VALID_CSV_PATH = "../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_validate.csv"


def _string_list_from_raw(raw_value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [line.strip() for line in str(parsed).splitlines() if line.strip()]
    except Exception:
        return [line.strip() for line in str(raw_value).splitlines() if line.strip()]


def build_training_text(row: Dict[str, str], k: int = 2) -> str:
    """构建RAG提示：优先使用本地检索的facts（若可用），否则回退旧格式。"""
    u1_personas = _string_list_from_raw(str(row.get("user 1 personas", "")))
    u2_personas = _string_list_from_raw(str(row.get("user 2 personas", "")))
    local_facts = [p for p in (u1_personas + u2_personas) if str(p).strip()]

    user_uttr = str(row.get("user_utterance", "")).strip()
    dialog_history = str(row.get("dialog_history", "")).strip()
    if not user_uttr:
        user_uttr = str(row.get("question", "")).strip()
    if not user_uttr:
        user_uttr = str(row.get("Best Generated Conversation", "")).strip()

    if local_facts and user_uttr:
        retr = BM25Retriever(local_persona_facts=local_facts)
        retrieved = retr.retrieve_top_k(user_uttr, k=k)
        numbered = [f"[P{i+1}] {retrieved[i]}" for i in range(len(retrieved))]
        assistant_ans = str(row.get("assistant_utterance", "")).strip() or str(row.get("answer", "")).strip()
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
            "<|im_start|>user\nHere are the persona facts:\n" + "\n".join(numbered)
        )
        if dialog_history:
            prompt += f"\n\n{dialog_history}\n"
        prompt += f"\nUser: {user_uttr}\n<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{assistant_ans}\n<|im_end|>\n"
        return prompt

    # 回退旧格式
    convo = str(row.get("Best Generated Conversation", ""))
    personas_section = " ".join([f"Persona 1: {p}" for p in u1_personas] + [f"Persona 2: {p}" for p in u2_personas])
    text = f"[CONTEXT] {personas_section} [DIALOGUE] {convo}"
    return text


def build_prompt(persona_context: list, history: str, question: str, answer_with_tags: str = None):
    """
    根据 HW1数据说明.pdf 构建 prompt 和 completion
    """
    # 1. 构建 Persona 上下文
    persona_str = "\n".join(persona_context)

    # 2. 构建 Prompt
    prompt = f"Your Personas:\n{persona_str}\n\n"
    prompt += f"Conversation History:\n{history}\n\n"
    prompt += f"User 2's response:\n{question}\n\n"  # User 2 是提问者
    prompt += "User 1's response:\n"  # 这是模型要开始生成的地方

    if answer_with_tags:
        # 如果提供了答案（用于训练），则构建 completion
        completion = f"{prompt}{answer_with_tags}"
        return completion
    else:
        # 否则只返回 prompt (用于推理)
        return prompt


def build_dataset(tokenizer, dataset_name):
    """
    重写 build_dataset 以符合 HW1数据说明.pdf 和 CSV 格式。
    - User 1 是 助手 (KB: user 1 personas)
    - User 2 是 用户 (提问者)
    - 模拟 TF-IDF RAG 来检索 Top-2 personas 并生成 [i] [j] 标签。
    """
    try:
        data = pd.read_csv(dataset_name).fillna("")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return Dataset.from_dict({"text": []})

    dataset_completions: List[str] = []
    print(f"Building dataset from {dataset_name}...")

    for ridx, (_, row) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Processing conversations")):
        kb_personas_str = str(row.get("user 1 personas", ""))
        kb_list = [p.strip() for p in kb_personas_str.split('\n') if p.strip()]
        if len(kb_list) < 2:
            continue

        conversation_lines = str(row.get("Best Generated Conversation", "")).split('\n')
        conversation_history: List[str] = []

        for i in range(len(conversation_lines) - 1):
            line = conversation_lines[i].strip()
            next_line = conversation_lines[i + 1].strip()
            if line.startswith("User 2:") and next_line.startswith("User 1:"):
                question = line.replace("User 2:", "").strip()
                answer_text = next_line.replace("User 1:", "").strip()
                history_context = "\n".join(conversation_history)
                try:
                    if not question:
                        continue
                    vectorizer = TfidfVectorizer().fit([question] + kb_list)
                    tfidf_matrix = vectorizer.transform([question] + kb_list)
                    question_vec = tfidf_matrix[0]
                    kb_vecs = tfidf_matrix[1:]
                    similarities = cosine_similarity(question_vec, kb_vecs)[0]
                    top_2_indices_0based = similarities.argsort()[-2:]
                    retrieved_personas = [kb_list[idx] for idx in top_2_indices_0based]
                    top_2_indices_1based = sorted([int(idx) + 1 for idx in top_2_indices_0based])
                    tags = " ".join(f"[{idx}]" for idx in top_2_indices_1based)
                    answer_with_tags = f"{answer_text} {tags}"
                    completion_str = build_prompt(
                        persona_context=retrieved_personas,
                        history=history_context,
                        question=question,
                        answer_with_tags=answer_with_tags,
                    )
                    dataset_completions.append(completion_str)
                    # 收集可视化样本（最多缓存100条）
                    if len(FINETUNE_PREVIEW) < 100:
                        FINETUNE_PREVIEW.append({
                            "row_index": ridx,
                            "kb_list": kb_list,
                            "retrieved_indices_1based": top_2_indices_1based,
                            "retrieved_personas": retrieved_personas,
                            "history": history_context,
                            "question": question,
                            "answer_text": answer_text,
                            "answer_with_tags": answer_with_tags,
                            "completion": completion_str,
                        })
                except Exception:
                    pass

                conversation_history.append(line)
                conversation_history.append(next_line)
            elif line.startswith("User 1:") or line.startswith("User 2:"):
                conversation_history.append(line)

    print(f"Finished building dataset. Total samples: {len(dataset_completions)}")
    dataset_final = Dataset.from_dict({"text": dataset_completions})
    print(f"Converted to Hugging Face Dataset object.")
    return dataset_final


def preprocess_function(examples, tokenizer, max_length: int):
    # datasets.map(batched=True) passes a dict-of-lists; rebuild row-wise samples.
    # When batched=False, a single example dict may be passed. Support both.
    rows: List[Dict[str, str]] = []
    if isinstance(examples, dict):
        # It can be a batched dict-of-lists or a single example (scalar values).
        first_val = next(iter(examples.values())) if examples else []
        if isinstance(first_val, list):
            length = len(first_val)
            for i in range(length):
                row_i = {k: (v[i] if isinstance(v, list) else v) for k, v in examples.items()}
                rows.append(row_i)
        else:
            rows.append(examples)  # single example
    else:
        # Fallback: coerce to a string field
        rows.append({"Best Generated Conversation": str(examples)})

    texts = [build_training_text(row) for row in rows]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=False,
    )
    return tokenized


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--train_csv_path", type=str, default=TRAIN_CSV_PATH)
    parser.add_argument("--valid_csv_path", type=str, default=VALID_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default="./my_finetuned_model")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=10)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--log_backend", type=str, choices=["tensorboard", "wandb", "none"], default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="hw1-baseline")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_best", action="store_true")
    # 预览/可视化
    parser.add_argument("--preview_n", type=int, default=0, help="打印前N条构造的伪RAG+标签样本")
    parser.add_argument("--preview_out", type=str, default=None, help="预览样本保存为JSONL文件路径")
    # 训练器选择
    parser.add_argument("--use_trl", action="store_true", help="使用 TRL 的 SFTTrainer（需已安装 trl）")
    # Tokenizer 复用/离线
    parser.add_argument("--tokenizer_path", type=str, default=None, help="从本地目录加载/保存 tokenizer（避免反复远端拉取）")
    parser.add_argument("--local_files_only", action="store_true", help="仅使用本地缓存文件加载（不访问网络）")
    args = parser.parse_args()
    # Set seed for reproducibility
    try:
        set_seed(args.seed)
    except Exception:
        pass

    # 使用新数据构建：从 CSV 生成带 prompt+completion 的文本样本
    train_text_dataset = build_dataset(tokenizer=None, dataset_name=args.train_csv_path)
    # 预览可视化
    if args.preview_n and args.preview_n > 0:
        print(f"[Preview] Show first {min(args.preview_n, len(FINETUNE_PREVIEW))} samples from constructed dataset...")
        for i, rec in enumerate(FINETUNE_PREVIEW[:args.preview_n], 1):
            print(f"\n==== Preview Sample #{i} ====")
            print(f"KB(size={len(rec.get('kb_list', []))}):")
            for idx, p in enumerate(rec.get("kb_list", []), 1):
                print(f"[{idx}] {p}")
            print("- Question:", rec.get("question", ""))
            print("- Retrieved (1-based idx):", rec.get("retrieved_indices_1based", []))
            print("- Retrieved personas:", rec.get("retrieved_personas", []))
            print("- History:\n", rec.get("history", ""))
            print("- Answer with tags:", rec.get("answer_with_tags", ""))
            print("- Completion (prompt+answer):\n", rec.get("completion", "")[:1000])
        if args.preview_out:
            try:
                out_dir = os.path.dirname(args.preview_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                with open(args.preview_out, "w", encoding="utf-8") as f:
                    for rec in FINETUNE_PREVIEW[:args.preview_n]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[Preview] Saved {min(args.preview_n, len(FINETUNE_PREVIEW))} samples to {args.preview_out}")
            except Exception as e:
                print(f"[Preview] Failed to save preview to {args.preview_out}: {e}")

    # Enable trust_remote_code for models like Qwen if requested
    trust_remote = args.trust_remote_code or ("qwen" in args.model_name.lower())
    # 优先从本地 tokenizer_path 复用；否则从模型名加载
    tokenizer_source = args.tokenizer_path if args.tokenizer_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote,
        local_files_only=bool(args.local_files_only)
    )
    # GPT-2 often needs an explicit pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 若指定了 tokenizer_path 且目录存在/可写，将当前 tokenizer 保存，便于后续直接离线复用
    if args.tokenizer_path:
        try:
            os.makedirs(args.tokenizer_path, exist_ok=True)
            tokenizer.save_pretrained(args.tokenizer_path)
        except Exception:
            pass

    model_load_kwargs = {"trust_remote_code": trust_remote}
    if args.bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)

    def tokenize_text(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_special_tokens_mask=False,
        )

    tokenized_dataset = None
    data_collator = None
    if not args.use_trl:
        tokenized_dataset = train_text_dataset.map(
            tokenize_text,
            batched=True,
            remove_columns=train_text_dataset.column_names,
            desc="Tokenizing",
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Prepare validation dataset if provided
    eval_dataset = None
    eval_text_dataset = None
    if args.valid_csv_path and os.path.exists(args.valid_csv_path):
        try:
            eval_text_dataset = build_dataset(tokenizer=None, dataset_name=args.valid_csv_path)
            if not args.use_trl:
                eval_dataset = eval_text_dataset.map(
                    tokenize_text,
                    batched=True,
                    remove_columns=eval_text_dataset.column_names,
                    desc="Tokenizing (valid)",
                )
        except Exception:
            eval_dataset = None
            eval_text_dataset = None

    # Configure logging backend
    report_to = []
    if args.log_backend == "tensorboard":
        report_to = ["tensorboard"]
    elif args.log_backend == "wandb":
        report_to = ["wandb"]
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # Build warmup kwargs: prefer warmup_steps when >0, else use warmup_ratio
    warmup_kwargs = {}
    if args.warmup_steps and args.warmup_steps > 0:
        warmup_kwargs["warmup_steps"] = args.warmup_steps
    elif args.warmup_ratio and args.warmup_ratio > 0:
        warmup_kwargs["warmup_ratio"] = args.warmup_ratio

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_steps=500,
        save_total_limit=1,
        fp16=False,
        bf16=bool(args.bf16),
        report_to=report_to,
        **warmup_kwargs,
    )

    # Logging and evaluation history containers
    training_logs: List[Dict] = []
    eval_logs: List[Dict] = []

    class LossLoggerCallback(TrainerCallback):
        """Capture training loss at each logging step for plotting later."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if "loss" in logs:
                training_logs.append({
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "loss": float(logs["loss"]),
                })

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            record = {
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }
            for k, v in metrics.items():
                try:
                    record[k] = float(v)
                except Exception:
                    pass
            # Derive perplexity if eval_loss is available
            if "eval_loss" in record:
                try:
                    record["perplexity"] = float(math.exp(record["eval_loss"]))
                except Exception:
                    pass
            eval_logs.append(record)

    class BestCheckpointCallback(TrainerCallback):
        """Save best checkpoint based on a metric (default: eval_loss)."""

        def __init__(self, output_dir: str, tokenizer, metric_name: str = "eval_loss", greater_is_better: bool = False):
            self.output_dir = output_dir
            self.tokenizer = tokenizer
            self.metric_name = metric_name
            self.greater_is_better = greater_is_better
            self.best_value = None
            self.best_dir = os.path.join(self.output_dir, "checkpoint-best")

        def _is_better(self, new, best):
            if best is None:
                return True
            return (new > best) if self.greater_is_better else (new < best)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics or self.metric_name not in metrics:
                return
            value = float(metrics[self.metric_name])
            if self._is_better(value, self.best_value):
                self.best_value = value
                os.makedirs(self.best_dir, exist_ok=True)
                # use attached trainer if available, fallback to model
                trainer = getattr(self, "trainer", None)
                model = kwargs.get("model", None)
                if trainer is not None:
                    trainer.save_model(self.best_dir)
                elif model is not None:
                    model.save_pretrained(self.best_dir)
                try:
                    self.tokenizer.save_pretrained(self.best_dir)
                except Exception:
                    pass
                # write best metrics
                try:
                    with open(os.path.join(self.best_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump({"metric": self.metric_name, "value": self.best_value, "global_step": int(state.global_step), "epoch": float(state.epoch) if state.epoch is not None else None}, f)
                except Exception:
                    pass

    @dataclass
    class EpochEvalEveryNCallback(TrainerCallback):
        """Trigger evaluation every N epochs via control.should_evaluate."""

        n: int = 10

        def on_epoch_end(self, args, state, control, **kwargs):
            try:
                current_epoch = int(state.epoch)
            except Exception:
                current_epoch = None
            if current_epoch is not None and self.n > 0 and current_epoch % self.n == 0:
                control.should_evaluate = True
            else:
                control.should_evaluate = False

    # 根据开关选择训练器
    if args.use_trl and SFTTrainer is not None:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_text_dataset,
            eval_dataset=eval_text_dataset,
            dataset_text_field="text",
        )
    else:
        if args.use_trl and SFTTrainer is None:
            print("[Warn] --use_trl 指定但未检测到 TRL；回退到 transformers.Trainer。")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # Register callbacks
    trainer.add_callback(LossLoggerCallback())
    trainer.add_callback(EpochEvalEveryNCallback(n=max(1, int(args.eval_every_n_epochs))))
    if args.save_best:
        best_cb = BestCheckpointCallback(output_dir=args.output_dir, tokenizer=tokenizer, metric_name="eval_loss", greater_is_better=False)
        # attach trainer for saving API
        best_cb.trainer = trainer
        trainer.add_callback(best_cb)

    trainer.train()

    # Ensure output dirs exist
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "training_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Save final model and tokenizer to a dedicated subdirectory
    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save logs to CSV
    try:
        import pandas as _pd
        if training_logs:
            _pd.DataFrame(training_logs).to_csv(os.path.join(logs_dir, "loss_history.csv"), index=False)
        if eval_logs:
            _pd.DataFrame(eval_logs).to_csv(os.path.join(logs_dir, "eval_history.csv"), index=False)
    except Exception:
        pass

    print(f"Model and tokenizer saved to {final_dir}")
    print(f"Training logs saved to {logs_dir}")


