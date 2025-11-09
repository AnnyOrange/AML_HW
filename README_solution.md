## AML 作业1: Personalized Language Models with RAG（解决方案指南）

本指南说明如何在本项目中完成从数据加载、检索评估、到微调与构建完整 RAG 流程的全部步骤。

### 目录结构
- `retriever.py`: 支持 BM25/TF-IDF 的“本地人设检索”器（仅在当前样本的人设列表上检索）。
- `evaluate_retrieval.py`: 基于“每行样本的本地人设”评估 Recall@K。
- `finetune.py`: 使用 `datasets` + `transformers` 微调因果语言模型；通过“伪 RAG（TF-IDF）+ [i][j] 标签”构建训练数据；统一 Prompt。
- `run.py`: 演示完整 RAG 流程（按对话动态构建KB → 检索 → 统一 Prompt → 生成），并打印 Ground Truth。
- `requirements.txt`: 项目依赖列表。

数据位于：`../Synthetic-Persona-Chat/data/`，包括 `Synthetic-Persona-Chat_train.csv`、`Synthetic-Persona-Chat_test.csv` 等。

### 统一规范（与作业要求一致）
- 统一数据格式：`finetune.py`、`run.py`、`retriever.py` 均直接解析 `Synthetic-Persona-Chat_*.csv` 的三列：
  - `user 1 personas`（助手/检索KB来源）、`user 2 personas`、`Best Generated Conversation`
- 统一角色：`User 1` 为助手（回答者），`User 2` 为用户（提问者）
- 统一 Prompt（训练与推理一致）：
  ```
  Your Personas:
  <persona 1>
  <persona 2>
  ...
  
  Conversation History:
  <历史多轮对话（原样拼接）>
  
  User 2's response:
  <当前用户提问>
  
  User 1's response:
  <模型在此开始生成；训练样本末尾拼接带标签答案>
  ```
- 按对话(per-dialogue)构建KB：`run.py` 对每行对话读取 `user 1 personas` 为 KB，仅在该 KB 内检索（Top-2）。

### 基本检索实现与引擎切换
- `retriever.py` 重构为统一接口：
  - `Retriever(kb_list, engine='bm25'|'tfidf')`
  - `retrieve(query, top_k)` 返回 `[(text, 1-based-index), ...]`
- 实现两个引擎：
  - BM25: `BM25Retriever(kb_list)` 使用 `rank_bm25`
  - TF-IDF: `TFIDFRetriever(kb_list)` 使用 `scikit-learn`
- `run.py` 支持 `--engine bm25|tfidf` 切换；每行对话动态创建检索器，Top-2 结果进入统一 Prompt。

### 环境准备
1) 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：`pandas`, `rank_bm25`, `scikit-learn`, `datasets`, `transformers`, `torch`, `accelerate`, `tqdm` 等。

### 运行步骤（按作业阶段）

#### 阶段一：微调（伪 RAG + 标签）
- 训练数据构造（已内置于 `finetune.py`）：
  - 使用 TF-IDF 在每场对话的 `user 1 personas` 上检索 Top-2
  - 将对应的人设文本作为 Persona 上下文进入 Prompt
  - 将答案拼接 `"[i] [j]"` 标签（i/j 为 1-based KB索引，排序保证稳定）
  - 将 `history` 为之前的多轮对话串（User 1/2 原样累加）
- 统一 Prompt/Completion（训练与推理一致）

运行示例（保存至 `./my_finetuned_model`，日志可用 TensorBoard/WandB）：

```bash
CUDA_VISIBLE_DEVICES=5 python finetune.py --log_backend tensorboard \
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

可选：使用 TRL 的 SFTTrainer

- 若已安装 `trl`，可以添加 `--use_trl` 开关，直接使用 `SFTTrainer` 进行 SFT（对齐/指令微调更友好）：
```bash
CUDA_VISIBLE_DEVICES=5 python finetune.py --use_trl \
  --train_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv \
  --valid_csv_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_valid.csv \
  --model_name Qwen/Qwen1.5-1.8B --trust_remote_code \
  --num_train_epochs 10 --eval_every_n_epochs 1 \
  --learning_rate 1e-5 --weight_decay 0.1 --max_grad_norm 1.0 --bf16 \
  --gradient_accumulation_steps 16 --warmup_ratio 0.03 --max_length 512 \
  --seed 42 --save_best --logging_steps 10 --output_dir ./runs/qwen_run_trl
```
- 若未安装 `trl` 但指定了 `--use_trl`，脚本会自动回退到 transformers.Trainer 并提示 Warning。

#### 阶段二：评估检索（Recall@2）

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

#### 阶段三：RAG 推理（按对话动态KB + 统一 Prompt）

```bash
python run.py --engine bm25 --top_k 2 --verbose \
  --kb_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_test.csv \
  --model_path ./runs/qwen_run-qwenchat/checkpoint-best
```

说明：
- `run.py` 会逐行读取 CSV（每行=一场对话），将该行 `user 1 personas` 作为 KB，按 `--engine` 检索 Top-2，构建统一 Prompt，调用本地微调模型生成，并打印：
  1) 问题（User 2）
  2) 目标检索（Ground Truth KB 提示：即“0) 知识库”）
  3) 目标输出（Ground Truth Answer = 下一句 User 1）
  4) 实际检索（Top-K，含 KB 中的 1-based 索引）
  5) 实际输出（模型生成）

（可选）切换 TF-IDF 引擎：

```bash
python run.py --engine tfidf --top_k 2 --verbose \
  --kb_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_test.csv \
  --model_path ./my_finetuned_model
```

### 注意事项
- 训练参数示例为快速演示，可根据硬件条件调整（序列长度、batch、累积步数、bf16/梯度检查点等）。
- 如果需要中文或更强的对话能力，请替换模型并相应修改 tokenizer/model 名称。

### 扩展命令与对比实验
1) 使用 rank_bm25 检索 Synthetic-Persona-Chat：
```bash
python run.py --engine bm25 --kb_path ../Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv --top_k 2 --verbose
```

2) 评估两种检索效果：
- BM25：
```bash
python evaluate_retrieval.py --top_k 2 --save_trace bm25_trace.json
```
- TF-IDF：修改评估脚本为 TF-IDF（或在评估脚本内增加引擎参数）。

3) 使用 Qwen 1.8B 微调（示例，需合适显存与依赖）：
```bash
python finetune.py --model_name Qwen/Qwen1.5-1.8B-Chat --trust_remote_code
```

4) 检索+微调 vs 直接问答对比：
- 检索增强：
```bash
python run.py --engine bm25 --top_k 2 --verbose --model_path ./my_finetuned_model
```
- 直接问答（不加检索上下文，可直接对用户问题做生成）——可临时修改 `run.py` 内 `final_prompt` 不包含 `[CONTEXT]` 部分，或编写独立脚本调用相同 `generator` 仅对 `user_query` 生成，比较回答质量。


