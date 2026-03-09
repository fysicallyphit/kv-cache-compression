from dotenv import load_dotenv
from openai import OpenAI
from semantic_search import semantic_search
import os

load_dotenv()

def generate_response():
    query, results = semantic_search()
    context = "\n\n".join(r["text"] for r in results)
    prompt = f"context: {context} \nprompt: {query}"
    client = OpenAI()
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = prompt)
    print(f"LLM response: \n{response.output_text}")

response = generate_response()
