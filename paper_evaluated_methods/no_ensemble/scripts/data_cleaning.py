import json

output_file = 'mmlu_dataset_pretty.json'

with open("/p/llmreliability/test_repos/no_ensemble/datasets/mmlu/test.json", "r") as file:
    data = json.load(file)
    
pretty_json = json.dumps(data, indent=4)
    
# Write the pretty-printed JSON to the output file
with open(output_file, 'w') as f:
    f.write(pretty_json)

print(f"Conversion complete. The pretty-printed JSON has been saved to '{output_file}'.")