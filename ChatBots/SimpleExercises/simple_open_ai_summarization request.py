"""
This is an exercise from maestro using openAI
using history
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import datetime

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MAX_SUMMARY_LENGTH = 20
MODEL = "gpt-4o-mini"
# MODEL = "gpt-4.1-mini"

def chat_with_ai(message:str) -> str:
    dt1 = datetime.datetime.now()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system",
                   "content": f"You task is to summarize the provided message in max {MAX_SUMMARY_LENGTH} words"},
                  {"role": "user", "content": message}],
        temperature=0.8,
        max_tokens=MAX_SUMMARY_LENGTH,
    )
    dt2 = datetime.datetime.now()
    print(dt2 - dt1)
    return response.choices[0].message.content

def main():
    history = ("Let's get your hands on a real OpenAI API call, step by step To start, I want to quickly check where "
               "your setup is: Are you working in your own local environment (like VS Code / terminal), or just inside "
               "the Maestro code editor right now? Do you already have an OpenAI API key created from the OpenAI dashboard? "
               "(Yes/No is enough — don’t paste the key.) Reply with: Local + Yes Local + No Maestro only + Yes Maestro only + No ")
    answer = chat_with_ai(history)
    print(answer)

if __name__ == "__main__":
    main()