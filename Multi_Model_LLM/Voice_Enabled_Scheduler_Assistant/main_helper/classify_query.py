"""
Query Classification:
Classify user queries as either “general” or “schedule-related” to determine the appropriate response mechanism.

main_helper/classify_query.py: Classifies user queries as “general” or "schedule-related".
"""

from openai import OpenAI
from main_helper.gpt_response import client

# Initialize the OpenAI client
client = OpenAI()


def classify_query(user_prompt):
    # Send the user prompt to the AI for classification
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Simply say 'schedule' if related to schedule and say 'general' if the question is a general question. Be brief."},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract the classification from the AI's response
    classification = completion.choices[0].message.content.strip().lower()
    return classification