import json
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

def load_aqua_dataset(file_path):
    """Load the AQUA dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def evaluate_model(model_name, dataset, num_samples=150):
    """Evaluate the model on the AQUA dataset and save results to a JSON file."""
    correct = 0
    total = min(num_samples, len(dataset))
    model_results = []

    # Randomly select 150 items from the dataset
    selected_samples = random.sample(dataset, total)

    # Sanitize the model name for use in a file name
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Use the specified output path
    output_dir = "/p/llmreliability/test_repos/no_ensemble/results/aqua"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the file path for incremental saving
    file_path = os.path.join(output_dir, f"{safe_model_name}_results.json")

    for sample in tqdm(selected_samples, desc=f"Evaluating {model_name}"):
        question = sample['question']
        options = sample['options']
        correct_answer = sample['correct']

        # Prepare the input for the model
        input_text = f"{question}\n\nOptions:\n" + "\n".join(options)
        input_text += "\n\nPlease provide your final answer as 'Final Answer: ' followed by the letter."

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                model=model_name,
            )
            response = chat_completion.choices[0].message.content

            # Save the question, options, the model's response, and the correct answer
            result = {
                "question": question,
                "options": options,
                "model_response": response,
                "correct_answer": correct_answer
            }
            model_results.append(result)

            # Incrementally save results after each iteration
            with open(file_path, 'w') as outfile:
                json.dump(model_results, outfile, indent=4)

            # Simple check if the correct answer is in the model's response
            if str(correct_answer) in response:
                correct += 1

            time.sleep(1)  # To avoid hitting API rate limits
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    accuracy = correct / total
    return accuracy, file_path

def main():
    dataset = load_aqua_dataset("/p/llmreliability/test_repos/no_ensemble/datasets/aqua/test.jsonl")
    models_to_evaluate = [
        "mistralai/Mixtral-8x22B-Instruct-v0.1"
        # "gpt-3.5-turbo",
        # "gpt-4o",
    ]

    results = {}
    for model in models_to_evaluate:
        accuracy, file_path = evaluate_model(model, dataset)
        results[model] = accuracy
        print(f"{model} accuracy: {accuracy:.2%}")
        print(f"Results saved to: {file_path}")

    print("\nFinal Results:")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.2%}")

if __name__ == "__main__":
    main()

    
   
