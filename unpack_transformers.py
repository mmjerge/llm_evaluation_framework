import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np 

model = AutoModelForCausalLM.from_pretrained("/scratch/mj6ux/Projects/llm_reliability/models/meta/base/meta-llama/Llama-2-7b-hf", 
                                             device_map="auto",
                                             torch_dtype=torch.float32, 
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("/scratch/mj6ux/Projects/llm_reliability/models/meta/base/meta-llama/Llama-2-7b-hf", 
                                          trust_remote_code=True)

tokenizer.pad_token_id = tokenizer.eos_token_id

inputs = tokenizer("What is the meaning of life?", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_new_tokens=200, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)

input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]

for token in generated_tokens:
    print(tokenizer.decode(token))

# result = outputs[:2]
# print(type(result))
# print(len(result))
# print(dir(outputs))
# print(hasattr(outputs, "loss"))
# print(hasattr(outputs, "logits"))

# print(dir(outputs.__getitem__))