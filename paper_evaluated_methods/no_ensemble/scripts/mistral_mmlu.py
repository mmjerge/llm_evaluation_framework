import os
import json
import time
from tqdm import tqdm
from mistralai import Mistral

# Set your Mistral API key
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

def load_mmlu_dataset(dataset):
    """Process the MMLU dataset from the provided JSON data."""
    processed_dataset = []
    for item in dataset:
        question = item['question'].strip()
        answer = item['answer']
        processed_dataset.append({
            "question": question,
            "answer": answer,
            "type": item.get('type', '')  # Include the 'type' if it exists
        })
    return processed_dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on the MMLU dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    for sample in tqdm(dataset[:total], desc=f"Evaluating {model_name}"):
        question = sample['question']
        correct_answer = sample['answer']

        try:
            chat_response = client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            )
            response = chat_response.choices[0].message.content

            # Save the question, the model's response, and the correct answer
            model_results.append({
                "question": question,
                "model_response": response,
                "correct_answer": correct_answer
            })

            # Check if the correct answer is in the model's response
            if correct_answer.lower() in response.lower():
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
    # Load the dataset from the provided JSON data
    with open("/p/llmreliability/test_repos/no_ensemble/datasets/mmlu/test.json", 'r') as f:
        dataset = json.load(f)
    
    processed_dataset = load_mmlu_dataset(dataset)
    
    models_to_evaluate = [
        "open-mixtral-8x22b",
    ]

    results = {}
    for model in models_to_evaluate:
        accuracy = evaluate_model(model, processed_dataset)
        results[model] = accuracy
        print(f"{model} accuracy: {accuracy:.2%}")

    print("\nFinal Results:")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.2%}")

if __name__ == "__main__":
    main()