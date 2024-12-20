import os
import json
import time
from tqdm import tqdm
from together import Together

def load_gsm8k_dataset(file_path):
    """Load the GSM8K dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on the GSM8K dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    # Initialize the Together client
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


    for sample in tqdm(dataset[:total], desc=f"Evaluating {model_name}"):
        question = sample['question']
        correct_answer = sample['answer']

        try:
            # Generate a response using the Together API
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": question}]
            )
            response = completion.choices[0].message.content

            # Save the question, the model's response, and the correct answer
            model_results.append({
                "question": question,
                "model_response": response,
                "correct_answer": correct_answer
            })

            # Simple check if the correct answer is in the model's response
            if str(correct_answer) in response:
                correct += 1

            time.sleep(1)  # Optional delay to avoid overloading the system

        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    accuracy = correct / total

    # Write the results to a JSON file
    with open(f"{model_name.replace('/', '_')}_results.json", 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    return accuracy

def main():
    dataset = load_gsm8k_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/gsm8k/test.jsonl")
    
    models_to_evaluate = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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

