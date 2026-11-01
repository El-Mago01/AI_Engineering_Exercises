from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

# Initialize the OpenAI client
# client = OpenAI()
client = OpenAI(api_key=api_key)

# Function to get a short model response
def get_gpt_response(user_input, user_name, ai_name):
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant to {user_name}. Name {ai_name}. You like to keep your messages short but friendly."},
            {"role": "user", "content": user_input},
        ]
    )
    return response.choices[0].message.content
    return response.choices[0].message['content']
