from sentence_transformers import SentenceTransformer, util
import json
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")


# Read and extract sentences from the JSON file
sentences = ["Eating watermelon seeds is generally safe and does not cause any harm to most people.",
             "Eating watermelon seeds is generally safe and can actually be healthy.",
             "Eating watermelon seeds is generally considered safe and does not pose any significant health risks.",
             "Watermelon seeds are not poisonous, but they can cause some side effects.",
             "The spiciest part of a chili pepper is the membrane, also known as the placenta, which holds the seeds.",
             "Placenta.",
             "Matadors wave red capes, also known as capotes or muletas, as part of their traditional attire and performance in bullfighting.",
             "Matadors wave red capes to attract and distract the bull during a bullfight.",
             "Humans use 100% of their brain, although not all areas are active at the same time.",
             "Contrary to a common myth, humans typically use 100% of their brain."]

print(sentences[:10])

# Encode all sentences to get their embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity between all pairs of embeddings
cos_sim = util.cos_sim(embeddings, embeddings)

# Prepare a list to hold all sentence pairs along with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim) - 1):
    for j in range(i + 1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

# Sort the list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

# Print the top-5 most similar sentence pairs
print("Top-5 most similar pairs:")
for score, i, j in all_sentence_combinations[:5]:
    print(f"{sentences[i]} \t {sentences[j]} \t {score.item():.4f}")