import os
from math import sqrt
from dotenv import load_dotenv
from openai import OpenAI
from output import nicer_print
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
documents = [
            #Semantic search works well
    "bolognese pasta with tomato meat sauce",
    "shwarma wrap with chicken and garlic sauce",
    "doner kebab with meat salad and flatbread",
    "sushi with rice seaweed",
    "margherita pizza with tomato mozzarella",
    "pad thai noodles with peanuts and lime",
    "Chicken curry with rice and spces",
    "ceasar salad with chicken lettuce and parmasan",
    "beef tacos with salsa and tortilla",
    "ramen noodle soup with egg",

    #exact-match examples
    "dish id dish-1001 bolognese",
    "dish id dish-1002 shawarma",
    "dish id dish-1003 doner kebab",
]

user_input = "I want some pasta"

def keyword_search(query, documents):
    query_words = set(query.split())
    results = []
    for doc in documents:
        score = len(query_words & set(doc.split()))
        if score > 0:
            results.append((doc, score))
    return sorted(
        results, key=lambda x: x[1], reverse=True
    )

def embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,

    )
    return [item.embedding for item in response.data]

def cosine(a,b):
    #cosine similarity
    #Calculates the distance between the vectors and multi-dimensional space
    dot=sum(x*y for x,y in zip(a,b))
    norm_a = sqrt(sum(x*x for x in a))
    norm_b = sqrt(sum(y*y for y in b))
    return dot/(norm_a*norm_b)

document_embeddings = embeddings(documents)

def semantic_search(query, top_k=3):
    query_embeddings = embeddings([query])[0]
    results = []
    for doc, embedding in zip(documents, document_embeddings):
        score=cosine(query_embeddings, embedding)
        results.append((doc, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

keyword_search_results = [
    doc for doc, score in keyword_search(user_input, documents)
]

semantic_search_results = [
    doc for doc, _score in semantic_search(user_input)
]

instructions = f""" 
You are a waiter, Recommend one dish to the guest. Dishes found by keyword search:
{chr(10).join(keyword_search_results)}
Dishes found by semantic search:
{chr(10).join(semantic_search_results)}
If nothing fits the request, say so. Answer in 2 sentences
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=user_input,
    instructions=instructions,
)

nicer_print(
    user_input,
    keyword_search_results,
    semantic_search_results,
    response.output_text
)