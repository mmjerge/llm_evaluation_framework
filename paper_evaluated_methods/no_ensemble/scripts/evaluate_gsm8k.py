import json
import os
import re

def extract_final_answer(answer):
    match = re.search(r'####\s*(\d+)', answer)
    return match.group(1) if match else None

def evaluate_responses(data):
    correct = 0
    total = len(data)

    for item in data:
        correct_answer = extract_final_answer(item['correct_answer'])
        model_response = item['model_response']

        if correct_answer is not None:
            if correct_answer in model_response:
                correct += 1
        else:
            print(f"Warning: Could not extract answer for a question in this dataset.")

    accuracy = correct / total if total > 0 else 0
    return accuracy

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return evaluate_responses(data)

def main():
    directory = 'results/gsm8k'  # Assuming the script is run from the parent directory
    results = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            model_name = filename.replace('_results.json', '')
            accuracy = process_json_file(file_path)
            results[model_name] = accuracy

    # Print results
    print("Accuracy results for each model:")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.2%}")

if __name__ == "__main__":
    main()