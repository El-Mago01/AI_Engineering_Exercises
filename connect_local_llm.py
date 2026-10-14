from ollama import Client

client = Client(host="192.168.1.75:11434")

def basic_chat():
    response=client.chat(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "Why is there a universe, and not nothing?"}]
    )
    print(response.message.content)

def main():
    basic_chat()

if __name__ == "__main__":
    main()