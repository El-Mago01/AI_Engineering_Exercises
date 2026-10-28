from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

MODEL="gpt-4o-mini"
# MODEL="text-embedding-3-small"
# Initialize OpenAI client with the API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

QUESTION = "what is the difference between an agent and a chatbot?"

    # Check for the clear command and either clear the file or process contents
with open("playground.txt", "r") as file:
    file_content = file.read()
if '//clear//' in file_content:
    open("playground.txt", "w").close()  # Clear the file
    quit()
messages=[]
if '//joke//' in file_content:
  open("playground.txt", "w")  # Clear the file

  messages=[
        # {"role": "system", "content": "You are a comedian, great at telling jokes."},
        {"role": "user", "content": QUESTION}
    ]

  temp = 0.9
  top_p=0.9
else:
  messages = [
  #            {"role": "system", "content": "You are an assistant."},
              {"role": "user", "content": QUESTION}
              ]
  temp = 0.3
  top_p = 0.1
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature = temp,
    max_tokens=300,
    top_p=top_p,
)
txt_response = response.choices[0].message.content
print(txt_response)
# Append the response to the file with a newline for clarity
# with open("playground.txt", "a") as f:
#     f.write("\n" + txt_response)
