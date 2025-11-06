## AML 作业1: Personalized Language Models with RAG（解决方案指南）

本指南说明如何在本项目中完成从数据加载、检索评估、到微调与构建完整 RAG 流程的全部步骤。

### 目录结构
- `retriever.py`: 基于 BM25 的“本地人设检索”器（仅在当前样本的人设列表上检索）。
- `evaluate_retrieval.py`: 基于“每行样本的本地人设”评估 Recall@K。
- `finetune.py`: 使用 `datasets` + `transformers` 微调因果语言模型，训练提示由“本地检索”构造。
- `run.py`: 演示完整 RAG 流程（检索 + 生成）。
- `requirements.txt`: 项目依赖列表。

数据位于：`../Synthetic-Persona-Chat/data/`，包括 `Synthetic-Persona-Chat_train.csv`、`Synthetic-Persona-Chat_test.csv` 等。

### 基本检索实现与引擎切换
- BM25 本地检索（`retriever.py`）
  - `BM25Retriever(local_persona_facts: List[str])`：仅对传入的“本地人设列表”建立索引与检索。
  - `retrieve_top_k(query, k)` / `retrieve_with_scores(...)`：返回本地 Top-K（及分数）。
- 引擎切换（`run.py`）
  - `--engine bm25`：演示本地检索；脚本会从传入 CSV 的第一行收集人设作为“本地facts”构建演示用检索器。
  - `--engine lightrag`：baseline LightRAG（需 `ZHIPUAI_API_KEY` 与 `book.txt`）。

### 环境准备
1) 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：`pandas`, `rank_bm25`, `scikit-learn`, `datasets`, `transformers`, `torch`, `accelerate` 等。

### 运行步骤

1) 微调模型（保存至 `./my_finetuned_model`，训练日志通过 TensorBoard/WandB 记录；默认模型：Qwen/Qwen1.5-1.8B-Chat）

```bash
# TensorBoard 日志（默认）
CUDA_VISIBLE_DEVICES=4 python finetune.py --log_backend tensorboard \
  --train_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv \
  --valid_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_valid.csv \
  --model_name Qwen/Qwen1.5-1.8B --trust_remote_code \
  --num_train_epochs 10 --eval_every_n_epochs 1 \
  --learning_rate 1e-5 --weight_decay 0.1 --max_grad_norm 1.0 --bf16 \
  --gradient_accumulation_steps 16 --warmup_ratio 0.03 --max_length 512 \
  --seed 42 --save_best --logging_steps 10 --output_dir ./runs/qwen_run

# 或使用 WandB 日志
python finetune.py --log_backend wandb --wandb_project hw1-baseline \
  --train_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv \
  --valid_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_valid.csv \
  --model_name Qwen/Qwen1.5-1.8B-Chat --trust_remote_code \
  --num_train_epochs 10 --eval_every_n_epochs 1 \
  --learning_rate 1e-5 --weight_decay 0.1 --max_grad_norm 1.0 --bf16 \
  --gradient_accumulation_steps 16 --warmup_ratio 0.03 --max_length 512 \
  --seed 42 --save_best --logging_steps 10
```

说明：
- 训练与验证数据路径通过 `--train_csv_path` 与 `--valid_csv_path` 指定（验证集为 `../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_validate.csv`）。
- 训练数据按“每行样本进行本地检索（k=2）”，将检索到的人设以 `[P1]`、`[P2]` 形式注入提示词（Qwen 聊天模版），缺列时回退旧格式。
- 使用 `Trainer` 训练，默认启用 TensorBoard 日志；可切换为 WandB；`--logging_steps 10` 记录 step 级训练曲线；`--save_best` 会按 `eval_loss` 额外保存 `checkpoint-best/`。
- 日志位置：
  - TensorBoard：`./my_finetuned_model/logs/`（可用 `tensorboard --logdir ./my_finetuned_model/logs` 查看）
  - WandB：在你的项目面板 `hw1-baseline` 中查看曲线与评估。
- 每 10 个 epoch 在验证集上自动评估（可用 `--eval_every_n_epochs` 调整）。

模型与检查点存储约定：
- 中间检查点：`OUTPUT_DIR/checkpoint-<step>/`
- 最优模型（需 `--save_best`）：`OUTPUT_DIR/checkpoint-best/`
- 最终模型（训练结束时保存）：`OUTPUT_DIR/final_checkpoint/`

备注（模型选择）：默认使用 `Qwen/Qwen1.5-1.8B-Chat`（需 `--trust_remote_code`）。
示例（并展示新参数）：
```bash
CUDA_VISIBLE_DEVICES=7 python finetune.py \
  --model_name Qwen/Qwen1.5-1.8B-Chat --trust_remote_code --log_backend tensorboard \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 --weight_decay 0.1 --max_grad_norm 1.0 --bf16 \
  --warmup_ratio 0.03 --max_length 512 \
  --train_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv \
  --valid_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_valid.csv \
  --num_train_epochs 10 --eval_every_n_epochs 1 \
  --save_best --seed 42 --logging_steps 10
```

参数说明（新增）：
- `--weight_decay`：权重衰减（L2 正则），建议 0.1。
- `--gradient_accumulation_steps`：梯度累计步数，用于增大有效 batch。
- `--max_grad_norm`：梯度裁剪上限，建议 1.0。
- `--warmup_steps`/`--warmup_ratio`：学习率预热；若设置了 `warmup_steps>0` 则优先生效，否则使用 `warmup_ratio`。
- `--max_length`：tokenize 的最大长度。
- `--seed`：随机种子，默认 42。
- `--save_best`：开启后按 `eval_loss` 保存最佳模型到 `checkpoint-best/`（不受 `save_total_limit` 清理影响）。
- `--logging_steps`：日志步频，建议 10（便于在 TensorBoard/WandB 查看 step 级 train/eval 曲线）。

2) 评估检索效果（Recall@K，基于“每行本地人设”）

```bash
# 评估 BM25 检索（保存追踪文件）
python evaluate_retrieval.py --top_k 2 --save_trace retrieval_trace.json

# 评估 LightRAG：建议自定义评估脚本或复用上面 BM25 的测试样本，对 rag.query 的结果进行命中统计
# （作业最低要求可先提交 BM25 的 Recall@K）
```

说明：
- 脚本已改为“逐行本地检索”：每行样本将合并该行两列 personas 作为本地事实，创建临时 BM25，仅在本地事实内检索。
- 若存在列 `user_utterance` 与 `assistant_utterance_grounded_on_persona_facts`，将分别作为查询与严格标签（按行内真实引用事实计算 Recall）。
- 否则回退到旧格式：从对话抽 QA，用启发式筛样本，再按子串重叠计算 Recall。
- 可用 `--top_k` 控制 K；`--save_trace` 会输出逐样本检索追踪 JSON。

3) 运行完整 RAG 流程示例（BM25 引擎，展示检索细节）

```bash
python run.py --engine bm25 --top_k 2 --verbose --kb_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv --model_path ./my_finetuned_model
```

说明：
- `bm25` 引擎为演示用途：从传入 CSV 第一行构造“本地 facts”并初始化 `BM25Retriever`；随后加载 `./my_finetuned_model`，完成检索+生成与可选的细节打印。

（可选）使用 baseline LightRAG 引擎

```bash
# 使用 ZHIPUAI_API_KEY 的 LightRAG，检索 book.txt
export ZHIPUAI_API_KEY=你的密钥
python run.py --engine lightrag --lightrag_path ./book.txt

# 使用 ZHIPUAI_API_KEY 的 LightRAG，递归检索 ../Synthetic-Persona-Chat/data 目录
python run.py --engine lightrag --lightrag_path ../Synthetic-Persona-Chat/data
```

说明：
- 需要设置环境变量 `ZHIPUAI_API_KEY`。
- 需要在 `hw1_baseline/` 目录下提供 `book.txt` 作为示例语料。

### 注意事项
- 训练参数示例为快速演示，可根据硬件条件调整（序列长度、batch、累积步数、bf16/梯度检查点等）。
- 如果需要中文或更强的对话能力，请替换模型并相应修改 tokenizer/model 名称。

### 扩展命令与对比实验
1) 使用 ZHIPUAI_API_KEY 的 LightRAG 检索：
```bash
export ZHIPUAI_API_KEY=你的密钥
# a) 检索 book.txt
python run.py --engine lightrag --lightrag_path ./book.txt
```

2) 使用 rank_bm25 检索 Synthetic-Persona-Chat：
```bash
python run.py --engine bm25 --kb_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv --top_k 2 --verbose
```

3) 评估两种检索效果：
- BM25：
```bash
python evaluate_retrieval.py --top_k 2 --save_trace bm25_trace.json
```
- LightRAG：可复用相同测试样本，统计 rag.query 返回的命中率（需自编评估脚本）。

4) 使用 Qwen 1.8B 微调（示例，需合适显存与依赖）：
```bash
python finetune.py --model_name Qwen/Qwen1.5-1.8B-Chat --trust_remote_code
```

5) 检索+微调 vs 直接问答对比：
- 检索增强：
```bash
python run.py --engine bm25 --top_k 2 --verbose --model_path ./my_finetuned_model
```
- 直接问答（不加检索上下文，可直接对用户问题做生成）——可临时修改 `run.py` 内 `final_prompt` 不包含 `[CONTEXT]` 部分，或编写独立脚本调用相同 `generator` 仅对 `user_query` 生成，比较回答质量。


