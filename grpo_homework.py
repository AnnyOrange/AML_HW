"""
AML HW2: GRPO Training for Mathematical Reasoning
=================================================

This is a simplified GRPO (Group Relative Policy Optimization) implementation
for training language models on mathematical reasoning tasks.

Instructions:
-------------
Fill in the 5 blanks marked with TODO. Each blank should be no more than 10 lines.
Make sure you understand the GRPO algorithm before filling in the blanks.
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# ============================================================================
# Part 1: Dataset Processing
# ============================================================================

class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""

    def __init__(self, split="train", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("./gsm8k", "main", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Extract the final numerical answer
        # Answer format: "#### 42"
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        # ========================================================================
        # TODO 1: Tokenize the prompt (3-5 lines)
        # ========================================================================

        # YOUR CODE HERE (3-5 lines)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # END TODO 1
        # ========================================================================

        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "prompt": prompt,
            "answer": answer_number,
        }

    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        # GSM8K answers end with #### followed by the answer
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        return None


# ============================================================================
# Part 2: Reward Function
# ============================================================================

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
    # Look for patterns like "The answer is 42" or "#### 42"
    patterns = [
        r'answer is\s*([-+]?\d+[\d,]*\.?\d*)',
        r'####\s*([-+]?\d+[\d,]*\.?\d*)',
        r'=\s*([-+]?\d+[\d,]*\.?\d*)(?:\s|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    return None


def compute_reward(completions, ground_truth_answers):
    """
    Compute binary rewards: 1 if correct, 0 if incorrect.

    Args:
        completions: List of completion strings
        ground_truth_answers: List or Tensor of ground truth answers

    Returns:
        rewards: Tensor of shape (batch_size,) with values 0 or 1 on appropriate device
    """
    rewards = []
    for completion, gt_answer in zip(completions, ground_truth_answers):
        gt_value = gt_answer.item() if isinstance(gt_answer, torch.Tensor) else gt_answer
        predicted = extract_answer_from_completion(completion)
        if predicted is not None and abs(predicted - gt_value) < 1e-3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    device = ground_truth_answers[0].device if isinstance(ground_truth_answers[0], torch.Tensor) else 'cpu'
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# ============================================================================
# Part 3: GRPO Algorithm Implementation
# ============================================================================

def compute_advantages_grpo(rewards, group_size=4):
    """
    Compute advantages using GRPO (Group Relative Policy Optimization).

    In GRPO, for each group of responses to the same prompt:
    - Advantage = reward - mean(group_rewards)

    This encourages the model to generate responses better than average.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_size: Number of responses per prompt

    Returns:
        advantages: Tensor of shape (batch_size,) containing advantages
    """
    # ========================================================================
    # TODO 2: Implement GRPO advantage computation (5-8 lines)
    # ========================================================================

    # YOUR CODE HERE (5-8 lines)
    if rewards.size(0) % group_size != 0:
        raise ValueError("Rewards size must be divisible by group_size")
    rewards = rewards.view(-1, group_size)
    group_means = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - group_means
    advantages = advantages.view(-1)

    # END TODO 2
    # ========================================================================

    return advantages


def compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=0.2):
    """
    Compute PPO-style clipped policy loss.

    Args:
        logprobs: Current policy log probabilities, shape (batch_size, seq_len)
        old_logprobs: Old policy log probabilities, shape (batch_size, seq_len)
        advantages: Advantages, shape (batch_size,)
        loss_mask: Mask for valid tokens, shape (batch_size, seq_len)
        clip_eps: Clipping epsilon for PPO

    Returns:
        loss: Scalar loss value
    """
    # ========================================================================
    # TODO 3: Implement PPO-style policy loss (6-10 lines)
    # ========================================================================

    # YOUR CODE HERE (6-10 lines)
    ratio = torch.exp(logprobs - old_logprobs)
    advantages = advantages.unsqueeze(1).expand_as(ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    masked_loss = policy_loss * loss_mask
    loss = masked_loss.sum() / loss_mask.sum()

    # END TODO 3
    # ========================================================================

    return loss


# ============================================================================
# Part 4: Training Loop
# ============================================================================

def generate_completions(model, tokenizer, input_ids, attention_mask,
                        max_new_tokens=256, temperature=1.0, num_samples=4):
    """
    Generate multiple completions for each prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token ids, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        num_samples: Number of samples per prompt

    Returns:
        all_outputs: Dictionary containing generated sequences and info
    """
    model.eval()

    # Repeat inputs for multiple samples
    batch_size = input_ids.shape[0]
    input_ids_repeated = input_ids.repeat_interleave(num_samples, dim=0)
    attention_mask_repeated = attention_mask.repeat_interleave(num_samples, dim=0)

    do_sample = temperature > 0.0
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask_repeated,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode completions
    completions = []
    for output in outputs:
        # Get only the generated part (exclude prompt)
        prompt_len = input_ids.shape[1]
        generated_ids = output[prompt_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return {
        "output_ids": outputs,
        "completions": completions,
        "prompt_length": input_ids.shape[1],
    }


def compute_logprobs_from_model(model, input_ids, attention_mask, requires_grad=False):
    if requires_grad:
        model.train()
    else:
        model.eval()
    context = torch.no_grad() if not requires_grad else torch.enable_grad()
    with context:
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logprobs = F.log_softmax(logits, dim=-1)
    shifted_logprobs = logprobs[..., :-1, :]
    labels = input_ids[..., 1:]
    gathered_logprobs = torch.gather(shifted_logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered_logprobs


def train_grpo(
    model,
    tokenizer,
    train_loader,
    optimizer,
    device,
    num_epochs=1,
    group_size=4,
    clip_eps=0.2,
    max_new_tokens=256,
    ppo_epochs=3,  # Number of PPO update epochs per batch
):
    """
    Main GRPO training loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        group_size: Number of samples per prompt for GRPO
        clip_eps: PPO clipping epsilon
        max_new_tokens: Maximum number of tokens to generate
    """

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]

            # ====================================================================
            # TODO 5: Implement the GRPO training step (8-10 lines)
            # ====================================================================
            
            # YOUR CODE HERE (8-10 lines for the main training logic)
            model.train()
            gen_outputs = generate_completions(model, tokenizer, input_ids, attention_mask, max_new_tokens=max_new_tokens, num_samples=group_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            completions = gen_outputs["completions"]
            output_ids = gen_outputs["output_ids"].to(device)
            attention_mask = (output_ids != tokenizer.pad_token_id).to(device)
            prompt_length = gen_outputs["prompt_length"]
            gt_answers = torch.tensor(answers, dtype=torch.float32, device=device).repeat_interleave(group_size)
            rewards = compute_reward(completions, gt_answers)
            old_logprobs = compute_logprobs_from_model(model, output_ids, attention_mask, requires_grad=False).detach()
            advantages = compute_advantages_grpo(rewards, group_size=group_size).to(device).detach()
            loss_mask = (torch.arange(old_logprobs.shape[1], device=device) >= (prompt_length - 1)).float().unsqueeze(0).expand_as(old_logprobs)
            for _ in range(ppo_epochs):
                logprobs = compute_logprobs_from_model(model, output_ids, attention_mask, requires_grad=True)
                loss = compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=clip_eps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # END TODO 5
            # ====================================================================

            # Logging
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "reward": f"{rewards.mean().item():.4f}",
                "avg_reward": f"{total_reward/num_batches:.4f}",
            })
            

        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")


def evaluate(model, tokenizer, split="test", batch_size=8, max_new_tokens=256, device=None):
    model.eval()
    dataset = GSM8KDataset(split=split, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in x],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in x],
                batch_first=True,
                padding_value=0
            ),
            "answer": [item["answer"] for item in x],
        }
    )
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = torch.tensor(batch["answer"], dtype=torch.float32, device=device)
            gen_outputs = generate_completions(model, tokenizer, input_ids, attention_mask, max_new_tokens=max_new_tokens, num_samples=1, temperature=0.0)
            completions = gen_outputs["completions"]
            rewards = compute_reward(completions, answers)
            total_correct += rewards.sum().item()
            total_samples += len(rewards)
    accuracy = total_correct / total_samples
    return accuracy


# ============================================================================
# Part 5: Main Function
# ============================================================================

def main():
    # Configuration
    if len(sys.argv) < 2:
        print("Usage: python grpo_homework.py <MODEL_PATH>")
        sys.exit(1)
    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2  # Temporarily reduce to 1 to avoid OOM; increase later if possible
    group_size = 4  # Number of samples per prompt
    num_epochs = 1
    learning_rate = 1e-6
    max_new_tokens = 128

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    print("Loading dataset...")
    train_dataset = GSM8KDataset(split="train[:100]", tokenizer=tokenizer)  # Use small subset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in x],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in x],
                batch_first=True,
                padding_value=0
            ),
            "prompt": [item["prompt"] for item in x],
            "answer": [item["answer"] for item in x],
        }
    )

    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting GRPO training...")
    train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    print("Evaluating on test set...")
    test_accuracy = evaluate(model, tokenizer, split="test", batch_size=32, max_new_tokens=max_new_tokens, device=device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    ckpt_dir = "./saved_model"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"模型已保存到 {ckpt_dir}")


if __name__ == "__main__":
    main()
