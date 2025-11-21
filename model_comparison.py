import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re

# Reuse functions from previous scripts
def extract_answer(answer_str):
    match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
    if match:
        return float(match.group(1).replace(',', ''))
    return None

def extract_answer_from_completion(completion):
    patterns = [
        r'answer is\s*([-+]?\d+[\d,]*\.?\d*)',
        r'####\s*([-+]?\d+[\d,]*\.?\d*)',
        r'=\s*([-+]?\d+[\d,]*\.?\d*)(?:\s|$)'
    ]
    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    return None

def generate_completion(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, device=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion

def main():
    baseline_path = "/data/xuzijing/Qwen2.5-1.5B-Instruct"
    trained_path = "/home/xzj/ml/HW2/saved_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_questions = 10  # Number of questions to compare

    print("Loading baseline model...")
    tokenizer_baseline = AutoTokenizer.from_pretrained(baseline_path)
    model_baseline = AutoModelForCausalLM.from_pretrained(baseline_path, torch_dtype=torch.bfloat16, device_map=device)

    print("Loading trained model...")
    tokenizer_trained = AutoTokenizer.from_pretrained(trained_path)
    model_trained = AutoModelForCausalLM.from_pretrained(trained_path, torch_dtype=torch.bfloat16, device_map=device)

    # Load test dataset
    test_data = load_dataset("./gsm8k", "main", split="test")[:num_questions]

    print("\nComparison of Model Responses (First 10 Test Questions):\n")
    print("{:<5} | {:<50} | {:<10} | {:<50} | {:<10} | {:<50} | {:<10}".format(
        "Idx", "Question (truncated)", "GT", "Baseline Response (truncated)", "Correct?", "Trained Response (truncated)", "Correct?"
    ))
    print("-" * 200)

    for idx, item in enumerate(test_data["question"]):
        question = item[:50] + "..." if len(item) > 50 else item
        gt_answer = extract_answer(test_data["answer"][idx])

        prompt = f"Question: {item}\nAnswer: Let's solve this step by step.\n"

        # Baseline generation
        baseline_completion = generate_completion(model_baseline, tokenizer_baseline, prompt, device=device)
        baseline_pred = extract_answer_from_completion(baseline_completion)
        baseline_correct = "Yes" if baseline_pred is not None and abs(baseline_pred - gt_answer) < 1e-3 else "No"
        baseline_resp = baseline_completion[:50] + "..." if len(baseline_completion) > 50 else baseline_completion

        # Trained generation
        trained_completion = generate_completion(model_trained, tokenizer_trained, prompt, device=device)
        trained_pred = extract_answer_from_completion(trained_completion)
        trained_correct = "Yes" if trained_pred is not None and abs(trained_pred - gt_answer) < 1e-3 else "No"
        trained_resp = trained_completion[:50] + "..." if len(trained_completion) > 50 else trained_completion

        print("{:<5} | {:<50} | {:<10} | {:<50} | {:<10} | {:<50} | {:<10}".format(
            idx, question, gt_answer, baseline_resp, baseline_correct, trained_resp, trained_correct
        ))

    md_file = "comparison.md"
    with open(md_file, "w") as f:
        f.write("# Model Comparison on GSM8K Test Questions\n\n")
        f.write("| Idx | Question (truncated) | GT | Baseline Response (truncated) | Correct? | Trained Response (truncated) | Correct? |\n")
        f.write("|-----|----------------------|----|-------------------------------|----------|------------------------------|----------|\n")

        for idx, item in enumerate(test_data["question"]):
            question = item[:50] + "..." if len(item) > 50 else item
            gt_answer = extract_answer(test_data["answer"][idx])

            prompt = f"Question: {item}\nAnswer: Let's solve this step by step.\n"

            # Baseline generation
            baseline_completion = generate_completion(model_baseline, tokenizer_baseline, prompt, device=device)
            baseline_pred = extract_answer_from_completion(baseline_completion)
            baseline_correct = "Yes" if baseline_pred is not None and abs(baseline_pred - gt_answer) < 1e-3 else "No"
            baseline_resp = baseline_completion[:50] + "..." if len(baseline_completion) > 50 else baseline_completion

            # Trained generation
            trained_completion = generate_completion(model_trained, tokenizer_trained, prompt, device=device)
            trained_pred = extract_answer_from_completion(trained_completion)
            trained_correct = "Yes" if trained_pred is not None and abs(trained_pred - gt_answer) < 1e-3 else "No"
            trained_resp = trained_completion[:50] + "..." if len(trained_completion) > 50 else trained_completion

            f.write(f"| {idx} | {question} | {gt_answer} | {baseline_resp} | {baseline_correct} | {trained_resp} | {trained_correct} |\n")

    print(f"Comparison saved to {md_file}")

if __name__ == "__main__":
    main()
