import json
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search():
    query = input("Search for legal and privacy data: ")
    embedded_query = model.encode(query)
    while True:
        try: 
            k = int(input("How many results would you like to return? "))
            break
        except ValueError:
            print("invalid input; must be an integer.")
    with open("outputs/embedded_chunks.json", "r") as f:
        data = json.load(f)
    for d in data:
        cos_similarity = cosine_similarity([embedded_query], [d["embedding"]])[0][0] #cosine_similarity returns 1x1 matrix of a score
        d["cos_score"] = cos_similarity.tolist()
    ranked = sorted(data, key=lambda data: data["cos_score"], reverse=True) 
    top_k = ranked[:k]
    # for rank, each_k in enumerate(top_k, start = 1):
    #     print(f"\nResult: {rank}")
    #     print(f"\nScore: {cos_similarity}")
    #     print(f"\nSource: {each_k["source"]}")
    #     print(f"\nText: {each_k["text"]}") 
    return query, top_k