from openai import OpenAI
from pet_support import FALLBACK
from dotenv import load_dotenv
import os
import datetime
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class OpenAIEmbedder:
    """Create embeddings for lore chunks and user questions."""

    def __init__(self):
        self.client = None

    def embed(self, texts):
        if self.client is None:
            self.client = OpenAI()
        dt1 = datetime.datetime.now()
        response = self.client.embeddings.create(
        model="text-embedding-3-small",
            input=texts,
        )
        dt2 = datetime.datetime.now()
        print(dt2 - dt1)
        print(texts)
        return [item.embedding for item in response.data]


class OpenAIGenerator:
    """Generate the final grounded DigiPet reply."""

    def __init__(self):
        self.client = None

    def generate(self, system_prompt, history, user_message):
        if self.client is None:
            self.client = OpenAI()

        messages = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            # messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_message})
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=messages,
                temperature=0.3,
            )
        return response.output_text.strip()
