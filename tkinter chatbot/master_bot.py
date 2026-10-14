from tkinter import NORMAL, DISABLED
import os
from setup import (
    build_interface,
    create_and_insert_user_frame,
    create_and_insert_bot_frame
)
from openai import OpenAI

# Initialize the OpenAI client with the API key
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def send(event=None):
    chat_window.config(state=NORMAL)

    user_input = user_entry_box.get("1.0", "end-1c").strip()
    print("The user provided: ", user_input)
    if user_input:
        bot_response = get_bot_response(user_input)
        print("The bot said: ", bot_response)
        create_and_insert_user_frame(chat_window, user_input)
        create_and_insert_bot_frame(chat_window, bot_response)

        user_entry_box.delete("1.0", "end")
        chat_window.see("end")

    chat_window.config(state=DISABLED)


# Generating response

# Generating response
def get_bot_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": user_input}
        ],
        max_tokens=150,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()


# Generating response
def get_bot_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": user_input}
        ],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


### END

base_window, chat_window, user_entry_box = build_interface(send)
base_window.mainloop()