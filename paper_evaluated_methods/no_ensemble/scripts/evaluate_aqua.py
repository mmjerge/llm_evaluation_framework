import json
import os
import re
from fuzzywuzzy import fuzz

def extract_answer(response, options):
    # Try to find a single letter answer (A, B, C, D, or E) at the beginning or end of the response
    match = re.search(r'(?:^|[^\w])(A|B|C|D|E)(?:$|[^\w])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for phrases like "The answer is A" or "Therefore, the correct answer is B."
    match = re.search(r'(?:answer|solution)\s+is\s*[:\s(]*([A-E])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Extract numerical values from the response
    numbers_in_response = re.findall(r'\d+(\.\d+)?', response)

    # Match extracted numbers against the options
    for number in numbers_in_response:
        for i, option in enumerate(options):
            if number in option:
                return chr(65 + i)  # Convert index to corresponding letter (0 -> A, 1 -> B, etc.)

    # Look for the answer expressed in words
    words_to_letters = {
        'A': ['A', 'FIRST', 'ONE'],
        'B': ['B', 'SECOND', 'TWO'],
        'C': ['C', 'THIRD', 'THREE'],
        'D': ['D', 'FOURTH', 'FOUR'],
        'E': ['E', 'FIFTH', 'FIVE']
    }
    
    for letter, words in words_to_letters.items():
        for word in words:
            if re.search(r'\b' + word + r'\b', response.upper()):
                return letter
    
    return None

def fuzzy_match(extracted, correct):
    return fuzz.ratio(extracted.lower(), correct.lower()) > 80

def evaluate_responses(data):
    correct = 0
    total = len(data)
    incorrect_samples = []
    extraction_failures = []
    
    for item in data:
        correct_answer = item['correct_answer'].upper()
        model_response = item['model_response']
        options = item.get('options', [])
        
        extracted_answer = extract_answer(model_response, options)
        
        if extracted_answer is None:
            extraction_failures.append({
                'question': item['question'],
                'correct_answer': correct_answer,
                'model_response': model_response[:200]
            })
            print(f"Failed to extract: {model_response[:100]}...")
        elif extracted_answer == correct_answer:
            correct += 1
            print(f"Correct: Extracted {extracted_answer}, Correct {correct_answer}")
        elif fuzzy_match(extracted_answer, correct_answer):
            correct += 1
            print(f"Fuzzy match: Extracted {extracted_answer}, Correct {correct_answer}")
        else:
            incorrect_samples.append({
                'question': item['question'],
                'correct_answer': correct_answer,
                'extracted_answer': extracted_answer,
                'model_response': model_response[:200]
            })
            print(f"Incorrect: Extracted {extracted_answer}, Correct {correct_answer}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, incorrect_samples, extraction_failures

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return evaluate_responses(data)

def main():
    directory = '/p/llmreliability/test_repos/no_ensemble/results/aqua'  # Assuming the script is run from the parent directory
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
    print("\nAccuracy results for each model on AQUA dataset:")
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
