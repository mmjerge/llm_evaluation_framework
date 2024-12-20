import os
import json
import time
from tqdm import tqdm
from mistralai import Mistral

# Set your Mistral API key
client = Mistral(
    api_key=os.environ.get("MISTRAL_API_KEY"),
)

def load_aqua_dataset(file_path):
    """Load the AQUA dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on the AQUA dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    for sample in tqdm(dataset[:total], desc=f"Evaluating {model_name}"):
        question = sample['question']  # Adjust based on the actual key in AQUA dataset
        correct_answer = sample['correct']  # Adjust based on the actual key in AQUA dataset

        try:
            chat_response = client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
            )

            response = chat_response.choices[0].message.content

            # Save the question, the model's response, and the correct answer
            model_results.append({
                "question": question,
                "model_response": response,
                "correct_answer": correct_answer
            })

            # Simple check if the correct answer is in the model's response
            if str(correct_answer) in response:
                correct += 1

            time.sleep(1)  # To avoid hitting API rate limits

        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    accuracy = correct / total

    # Write the results to a JSON file
    with open(f"{model_name}_results.json", 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    return accuracy

def main():
    dataset = load_aqua_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/aqua/test.jsonl")
    
    models_to_evaluate = [
        "open-mixtral-8x22b",
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
