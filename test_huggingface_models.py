from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import tqdm
import json
from utils.util import *

NEO4J_URI = "neo4j+s://023fca8e.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "XopDQ_W2ToQfo9fdgob8Zx4-piYAOK-qt26Ses0CBK0"

llama2 = LlamaBaseAgent("llama-base", 
                        neo4j_uri=NEO4J_URI, 
                        neo4j_username=NEO4J_USERNAME, 
                        neo4j_password=NEO4J_PASSWORD)

llama2large = LlamaLargeAgent("llama-large",
                              neo4j_uri=NEO4J_URI, 
                              neo4j_username=NEO4J_USERNAME, 
                              neo4j_password=NEO4J_PASSWORD)

mistral = MistralAgent("mistral-base",
                       neo4j_uri=NEO4J_URI, 
                       neo4j_username=NEO4J_USERNAME, 
                       neo4j_password=NEO4J_PASSWORD)

phi2 = MicrosoftAgent("phi-2",
                    neo4j_uri=NEO4J_URI, 
                    neo4j_username=NEO4J_USERNAME, 
                    neo4j_password=NEO4J_PASSWORD)

agents = [llama2large]

llama_model_path = "/p/llmreliability/llm_reliability/models/meta/base/meta-llama/Llama-2-7b-hf"

llama_large_model_path = "/p/llmreliability/llm_reliability/models/meta/large/meta-llama/Llama-2-70b-hf"

mistral_model_path = "/p/llmreliability/llm_reliability/models/mistral/base/mistralai/Mistral-7B-v0.1"

phi2_model_path = "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2"

questions = read_questions("/p/llmreliability/llm_reliability/data/truthful_qa/generation/validation-00000-of-00001.parquet")

qa_dict = {}
for question in questions:
    responses = {}
    for agent in agents:
        agent_name = agent.model_name
        if agent_name == "llama-base":
            responses[agent_name] = agent.generate_text(question, llama_model_path)
        elif agent_name == "llama-large":
            responses[agent_name] = agent.generate_text(question, llama_large_model_path)
        elif agent_name == "mistral-base":
            responses[agent_name] = agent.generate_text(question, mistral_model_path)
        elif agent_name == "phi-2":
            responses[agent_name] = agent.generate_text(question, phi2_model_path)
        else:
            pass
    qa_dict[question] = responses
print("QA complete.")

file_path = 'llama_large_agent_responses.json'
with open(file_path, 'w') as file:
        json.dump(qa_dict, file, indent=4)