import json

def extract_final_answer(response):
    # Find the last occurrence of "Final Answer:" in the response
    final_answer_index = response.rfind("Final Answer:")
    if final_answer_index != -1:
        # Extract the part after "Final Answer:"
        final_answer = response[final_answer_index + len("Final Answer:"):].strip()
        # Extract just the letter (assuming it's always a single letter)
        return final_answer[0]
    return None

def evaluate_accuracy(data):
    correct = 0
    total = len(data)

    for item in data:
        model_answer = extract_final_answer(item['model_response'])
        correct_answer = item['correct_answer']

        if model_answer == correct_answer:
            correct += 1

    accuracy = correct / total
    return accuracy

# Load the JSON data
with open('/p/llmreliability/test_repos/no_ensemble/results/aqua/mistralai_Mixtral-8x22B-Instruct-v0.1_results.json', 'r') as f:
    data = json.load(f)

# Calculate accuracy
accuracy = evaluate_accuracy(data)

print(f"Total questions: {len(data)}")
print(f"Correct answers: {int(accuracy * len(data))}")
print(f"Accuracy: {accuracy:.2%}")

# Print incorrect answers for analysis
print("\nIncorrect Answers:")
for i, item in enumerate(data, 1):
    model_answer = extract_final_answer(item['model_response'])
    correct_answer = item['correct_answer']
    if model_answer != correct_answer:
        print(f"Question {i}:")
        print(f"  Model's answer: {model_answer}")
        print(f"  Correct answer: {correct_answer}")
        print(f"  Question: {item['question'][:100]}...")  # Print first 100 characters of the question
        print()