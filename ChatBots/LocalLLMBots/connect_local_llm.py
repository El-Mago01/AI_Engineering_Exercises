from ollama import Client

QUESTION = "what is the difference between an agent and a chatbot?"
client = Client(host="192.168.1.75:11434")
# MODEL = "gemma4:e2b"
# MODEL = "hf.co/bartowski/gemma-1.1-2b-it-GGUF:latest"
MODEL = "qwen3:8b"
# MODEL = "nomic-embed-text:latest"

def basic_chat():
    response=client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}]
    )
    print(response.message.content)

def main():
    basic_chat()

if __name__ == "__main__":
    main()