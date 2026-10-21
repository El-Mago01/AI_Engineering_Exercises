from tkinter import *
import tkinter.scrolledtext as tks
from datetime import datetime


def create_and_insert_user_frame(chat_window, user_input):
    user_frame = Frame(chat_window, bg="#d0ffff")

    Label(
        user_frame,
        text=user_input,
        font=("Arial", 11),
        bg="#d0ffff"
    ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

    Label(
        user_frame,
        text=datetime.now().strftime("%H:%M"),
        font=("Arial", 7),
        bg="#d0ffff"
    ).grid(row=1, column=0, sticky="w")

    chat_window.insert("end", "\n ", "tag-right")
    chat_window.window_create("end", window=user_frame)


def create_and_insert_bot_frame(chat_window, bot_response):
    bot_frame = Frame(chat_window, bg="#ffffd0")

    Label(
        bot_frame,
        text=bot_response,
        font=("Arial", 11),
        bg="#ffffd0",
        wraplength=550,
        justify="left"
    ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

    Label(
        bot_frame,
        text=datetime.now().strftime("%H:%M"),
        font=("Arial", 7),
        bg="#ffffd0"
    ).grid(row=1, column=0, sticky="w")

    chat_window.insert("end", "\n ", "tag-left")
    chat_window.window_create("end", window=bot_frame)
    chat_window.insert(END, "\n\n")


def build_interface(send_callback):
    base_window = Tk()
    base_window.title("The Master Bot")
    base_window.geometry("600x350")

    chat_window = tks.ScrolledText(base_window, font="Arial")
    chat_window.tag_configure("tag-left", justify="left")
    chat_window.tag_configure("tag-right", justify="right")
    chat_window.config(state=DISABLED)

    send_button = Button(
        base_window,
        font=("Verdana", 12, "bold"),
        text="Send",
        bg="#fd94b4",
        activebackground="#ff467e",
        fg="#ffffff",
        command=send_callback
    )

    user_entry_box = Text(base_window, bd=1, bg="white", width=38, font="Arial")

    chat_window.place(x=1, y=1, height=300, width=600)
    user_entry_box.place(x=3, y=300, height=30, width=480)
    send_button.place(x=490, y=300, height=30, width=100)
    return base_window, chat_window, user_entry_box


if __name__ == "__main__":
    def placeholder_send(event=None):
        pass