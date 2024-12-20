import json
import re

def extract_answer(response, options):
    # Try to find a letter (A-E) followed by a closing parenthesis
    letter_match = re.search(r'\b([A-E])\)', response)
    if letter_match:
        return letter_match.group(1)

    # Try to find the exact text of one of the options
    for idx, option in enumerate(options):
        if option in response:
            return chr(65 + idx)  # Convert 0-4 to A-E

    # Try to find a number that could correspond to an option index
    number_match = re.search(r'\b([1-5])\b', response)
    if number_match:
        number = int(number_match.group(1))
        if 1 <= number <= len(options):
            return chr(64 + number)  # Convert 1-5 to A-E

    return None

def calculate_accuracy(data):
    total_questions = len(data)
    correct_answers = 0

    for item in data:
        question = item['question']
        model_response = item['model_response']
        correct_answer = item['correct_answer']
        options = item['options']

        # Extract the answer from the model's response
        extracted_answer = extract_answer(model_response, options)
        
        # Check if the extracted answer matches the correct answer
        is_correct = extracted_answer == correct_answer
        if is_correct:
            correct_answers += 1
        
        # Print question details
        print(f"Question: {question}")
        print(f"Model Response: {model_response[:100]}...")
        print(f"Extracted Answer: {extracted_answer}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Options: {options}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
        print("-" * 50)

    accuracy = (correct_answers / total_questions) * 100
    return accuracy

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# File path for the JSON file
json_file_path = '/p/llmreliability/test_repos/no_ensemble/scripts/meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_results.json'

# Load data from JSON file
dataset = load_data_from_json(json_file_path)

# Calculate and print the accuracy
accuracy = calculate_accuracy(dataset)
print(f"Accuracy: {accuracy:.2f}%")