from openai import OpenAI
import os
# Initialize OpenAI client with the API key
client = OpenAI()

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
        {"role": "system", "content": "You are a comedian, great at telling jokes."},
        {"role": "user", "content": "Tell a hilarious joke within 260 words"}
    ]

  temp = 0.9
  top_p=0.9
else:
  messages = [{"role": "system", "content": "You are an assistant."},
              {"role": "user", "content": file_content}
              ]
  temp = 0.3
  top_p = 0.1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature = temp,
    max_tokens=300,
    top_p=top_p,
)
txt_response = response.choices[0].message.content
# Append the response to the file with a newline for clarity
with open("playground.txt", "a") as f:
    f.write("\n" + txt_response)
