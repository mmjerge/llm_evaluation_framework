from openai import OpenAI
import json
import os
import time
from tqdm import tqdm

# Set your OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def load_mmlu_dataset(file_path):
    """Load the MMLU dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Check if data is a list (handle list of questions in a single line)
            if isinstance(data, list):
                for item in data:
                    question = item['question']
                    answer = item['answer']
                    dataset.append({
                        "question": question,
                        "answer": answer
                    })
            else:
                question = data['question']
                answer = data['answer']
                dataset.append({
                    "question": question,
                    "answer": answer
                })
    return dataset

def evaluate_model(model_name, dataset, num_samples=100):
    """Evaluate the model on the MMLU dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    for sample in tqdm(dataset[:total], desc=f"Evaluating {model_name}"):
        question = sample['question']
        correct_answer = sample['answer']

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                model=model_name,
            )

            response = chat_completion.choices[0].message.content

            # Save the question, the model's response, and the correct answer
            model_results.append({
                "question": question,
                "model_response": response,
                "correct_answer": correct_answer
            })

            # Simple check if the correct answer is in the model's response
            if correct_answer.strip() in response:
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
    dataset = load_mmlu_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/mmlu/test.json")
    
    models_to_evaluate = [
        "gpt-3.5-turbo",
        "gpt-4"
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


