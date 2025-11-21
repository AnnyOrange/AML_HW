import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re

class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""

    def __init__(self, split="test", tokenizer=None, max_length=512):
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
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "prompt": prompt,
            "answer": answer_number,
        }

    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
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

def generate_completions(model, tokenizer, input_ids, attention_mask, max_new_tokens=256, temperature=1.0, num_samples=1):
    model.eval()

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

    completions = []
    for output in outputs:
        prompt_len = input_ids.shape[1]
        generated_ids = output[prompt_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return {
        "output_ids": outputs,
        "completions": completions,
        "prompt_length": input_ids.shape[1],
    }

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python baseline_test.py <MODEL_PATH>")
        sys.exit(1)
    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("Evaluating baseline on test set...")
    test_accuracy = evaluate(model, tokenizer, split="test", batch_size=32, max_new_tokens=128, device=device)
    print(f"Baseline Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
