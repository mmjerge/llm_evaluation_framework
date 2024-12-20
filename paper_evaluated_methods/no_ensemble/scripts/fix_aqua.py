import json
import jsonlines

def load_questions(jsonl_file):
    questions = {}
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            questions[obj['question']] = obj['options']
    return questions

def update_results(results_file, questions):
    with open(results_file, 'r') as f:
        results = json.load(f)

    for result in results[:100]:
        question = result['question']
        if question in questions:
            result['options'] = questions[question]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    jsonl_file = '/p/llmreliability/test_repos/no_ensemble/datasets/aqua/test.jsonl'
    results_file = '/p/llmreliability/test_repos/no_ensemble/results/aqua/meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_results.json'

    questions = load_questions(jsonl_file)
    update_results(results_file, questions)
    print("Options have been appended to the results file.")

if __name__ == "__main__":
    main()