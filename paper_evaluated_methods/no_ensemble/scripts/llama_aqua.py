import os
import json
import time
import random
from tqdm import tqdm
from together import Together

# Initialize the Together client
client = Together()

def load_aqua_dataset(file_path):
    """Load the AQUA dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on random samples from the AQUA dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []
    
    # Take random samples
    random_samples = random.sample(dataset, total)
    
    for sample in tqdm(random_samples, desc=f"Evaluating {model_name}"):
        question = sample['question']
        options = sample['options']
        correct_answer = sample['correct']
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{question}\n\nOptions: {options}\n\nPlease output your final answer as the LETTER from the multiple choice options.",
                    }
                ],
                model=model_name,
            )
            response = chat_completion.choices[0].message.content
            
            # Save the question, the model's response, and the correct answer
            model_results.append({
                "question": question,
                "options": options,
                "model_response": response,
                "correct_answer": correct_answer,
                "answered_correctly": str(correct_answer) in response
            })
            
            # Simple check if the correct answer is in the model's response
            if str(correct_answer) in response:
                correct += 1
            
            time.sleep(1)  # To avoid hitting API rate limits
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    accuracy = correct / total

    # Write the results to a JSON file with improved formatting
    with open(f"{model_name.replace('/', '_')}_results.json", 'w') as outfile:
        json.dump(model_results, outfile, indent=4, ensure_ascii=False)

    return accuracy

def main():
    dataset = load_aqua_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/aqua/test.jsonl")
    models_to_evaluate = [
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ]
    
    results = {}
    for model in models_to_evaluate:
        accuracy = evaluate_model(model, dataset)
        results[model] = accuracy
        print(f"{model} accuracy: {accuracy:.2%}")

    print("\nFinal Results:")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.2%}")

if __name__ == "__main__":
    main()

