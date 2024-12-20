import json
import os
import re

def extract_answer(response):
    # Try to find a single letter answer (a, b, c, or d) at the beginning of the response
    match = re.match(r'^[(\s]*([a-d])[)\s.]', response.lower())
    if match:
        return match.group(1)
    
    # Look for phrases like "The correct answer is (a)" or "The answer is b."
    match = re.search(r'(?:correct\s+answer|answer)\s+is\s*[:\s(]*([a-d])', response.lower())
    if match:
        return match.group(1)
    
    # Look for the first occurrence of (a), (b), (c), or (d) in the response
    match = re.search(r'\(([a-d])\)', response.lower())
    if match:
        return match.group(1)
    
    # Look for a, b, c, or d followed by a period or parenthesis
    match = re.search(r'\b([a-d])[.)]', response.lower())
    if match:
        return match.group(1)
    
    # If no clear answer is found, return None
    return None

def compare_answers(extracted, correct):
    return extracted.lower() == correct.lower()

def evaluate_responses(data):
    correct = 0
    total = len(data)
    incorrect_samples = []
    extraction_failures = []
    
    for item in data:
        correct_answer = item['correct_answer'].lower()
        model_response = item['model_response']
        
        extracted_answer = extract_answer(model_response)
        
        if extracted_answer is None:
            extraction_failures.append({
                'question': item['question'],
                'correct_answer': correct_answer,
                'model_response': model_response[:200]
            })
        elif compare_answers(extracted_answer, correct_answer):
            correct += 1
        else:
            incorrect_samples.append({
                'question': item['question'],
                'correct_answer': correct_answer,
                'extracted_answer': extracted_answer,
                'model_response': model_response[:200]
            })

    accuracy = correct / total if total > 0 else 0
    return accuracy, incorrect_samples, extraction_failures

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return evaluate_responses(data)

def main():
    directory = '/p/llmreliability/test_repos/no_ensemble/results/mmlu'  # Assuming the script is run from the parent directory
    results = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            model_name = filename.replace('_results.json', '')
            accuracy, incorrect_samples, extraction_failures = process_json_file(file_path)
            results[model_name] = {
                'accuracy': accuracy,
                'incorrect_samples': incorrect_samples,
                'extraction_failures': extraction_failures
            }

    # Print results
    print("\nAccuracy results for each model on MMLU dataset:")
    for model, data in results.items():
        print(f"{model}: {data['accuracy']:.2%}")
        
    # Print some incorrect samples and extraction failures for manual verification
    for model, data in results.items():
        print(f"\n{model}:")
        print("Incorrect samples:")
        for sample in data['incorrect_samples'][:3]:
            print(f"Q: {sample['question'][:100]}...")
            print(f"Correct: {sample['correct_answer']}")
            print(f"Extracted: {sample['extracted_answer']}")
            print(f"Response: {sample['model_response']}")
            print("----")
        print("Extraction failures:")
        for sample in data['extraction_failures'][:3]:
            print(f"Q: {sample['question'][:100]}...")
            print(f"Correct: {sample['correct_answer']}")
            print(f"Response: {sample['model_response']}")
            print("----")

if __name__ == "__main__":
    main()