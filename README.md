# AML 作业 2：用于数学推理的 GRPO 训练

## 概述
本次作业的目标是实现 **GRPO（Group Relative Policy Optimization，组相对策略优化）**，这是一种专门为大语言模型设计的强化学习算法，能在数学推理任务上取得很好的效果。GRPO是PPO（Proximal Policy Optimization）的简化变体，已在多种前沿的推理模型（如 DeepSeek-R1）中取得优异表现。

## 背景

### GRPO 算法简介
GRPO 的主要特点：
1. **组内优势计算**：不使用价值网络，而是通过比较同一组响应的平均奖励来计算优势值；
2. **简单高效**：不需要critic网络，计算和代码实现都更轻量；
3. **效果显著**：已在多种顶尖 LLM 推理模型中验证有效。

### 算法流程
每一步训练包括：
1. **为每个提示采样多组响应**（例如 group_size=4）  
2. **计算所有响应的奖励**  
3. **计算组相对优势**
4. **使用 PPO 风格的目标更新策略（policy）**

---

## 作业任务
你需要在 `grpo_homework.py` 中完成 **5 个 TODO**：

### TODO 1：Tokenization（3-5行）
**位置**：`GSM8KDataset.__getitem__`  
**任务**：用 tokenizer 将输入 prompt 转换为模型可处理的张量。  

---

### TODO 2：GRPO 优势计算（5-8行）
**位置**：`compute_advantages_grpo`  
**任务**：实现核心的组相对优势计算。  

---

### TODO 3：PPO 策略损失（6-10行）
**位置**：`compute_policy_loss`  
**任务**：实现 PPO 截断目标损失。  

---

### TODO 4：计算模型 log 概率（6-8行）
**位置**：`compute_logprobs_from_model`  
**任务**：为生成的序列计算 log 概率。  

---

### TODO 5：GRPO 训练步骤实现（8-10行）
**位置**：`train_grpo`  
**任务**：实现完整的 GRPO 训练流程。  

---

## 环境配置
1. **配置环境**：
```bash
pip install -r requirements.txt
```
2. **模型路径**：可下载并使用 Qwen-2.5-1.5B 模型：
```
https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct
```
3. **GPU 配置**：默认使用 GPU：
```python
device = torch.device("cuda")
```

---

## 运行代码
测试你的实现：
```bash
python grpo_homework.py {YOUR_MODEL_PATH}
```
输出示例：
```
Loading tokenizer and model...
Loading dataset...
Setting up optimizer...
Starting GRPO training...
...
Training completed!
```

---

## 提交说明

**提交内容**：
1. 一份报告，描述：
- 算法及实现概述
- 训练参数设置
- RL训练过程记录
- 训练前后回答结果对比

2. 完成的 `grpo_homework.py`  
命名格式：
```
学号_姓名_hw2.py
```
例如：`12345678_zhangsan_hw2.py`  

---

## 评分标准
总分 100 分：
- **TODO 1** (15分)：正确实现 Tokenization
- **TODO 2** (15分)：正确计算优势
- **TODO 3** (15分)：正确实现 PPO 策略损失
- **TODO 4** (15分)：正确计算 log 概率
- **TODO 5** (20分)：完整训练流程，相对baseline有明显提升
- **报告** (20分)

---

## 资源链接
- [GRPO 原论文](https://arxiv.org/abs/2402.03300)  
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [AReaL 仓库](https://github.com/inclusionAI/AReaL)