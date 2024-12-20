import os
import json
import time
from tqdm import tqdm
from together import Together

# Initialize the Together client
client = Together()

def load_svamp_dataset(file_path):
    """Load the SVAMP dataset from a JSON file."""
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on the SVAMP dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    for sample in tqdm(dataset[:total], desc=f"Evaluating {model_name}"):
        body = sample['Body']
        question = sample['Question']
        correct_answer = sample['Answer']
        
        # Combine Body and Question
        full_question = f"{body} {question}"

        try:
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": full_question,
                    }
                ],
            )

            response = chat_completion.choices[0].message.content

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
    with open(f"{model_name.replace('/', '_')}_svamp_results.json", 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    return accuracy

def main():
    dataset = load_svamp_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/svamp/SVAMP.json")
    
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
