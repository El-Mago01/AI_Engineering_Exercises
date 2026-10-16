import re

from openai_adapters import OpenAIEmbedder, OpenAIGenerator
from pet_support import FALLBACK, MIN_SCORE, TOP_K
from pet_support import is_stats_question, similarity_score
import os


class PetChat:
    """Own the lore index, retrieval, grounding, generation, and history."""

    def __init__(self, base_dir, embedder=None, generator=None):
        self.base_dir = base_dir
        self.embedder = embedder or OpenAIEmbedder()
        self.generator = generator or OpenAIGenerator()
        self.records = self.load_lore(base_dir / "lore")
        self.history = []

    def __discover_files(self, folder):
        """Return supported source files in a predictable order."""
        supported_files = []

        for path in folder.iterdir():
            if path.is_file() and path.suffix in [".md", ".txt"]:
                supported_files.append(path)

        return sorted(supported_files)

    def __create_combined_list(self, meta_data:list, vector_data:list)->list:
        if len(meta_data) != len(vector_data):
            raise ValueError("Length of meta_data and vector_data do not match.")
        chunks = []
        index = 0
        for id, chunk, filename in meta_data:
            print (index, id, chunk, filename)
            chunk = {
                "record_id": id,
                "text": chunk,
                "source": filename,
                "embedding": vector_data[index],
            }
            chunks.append(chunk)
            index += 1
        return chunks


    def load_lore(self, lore_dir):
        """Load lore files and attach embeddings to searchable records."""
        lore_files = self.__discover_files(lore_dir)
        total_text = ""
        id = 0
        chunks = []
        for lore_file in lore_files:
            with open(lore_file, "r", encoding="UTF-8") as f:
                file_name = os.path.basename(f.name)
                total_text += f.read()
                text_chunks = total_text.strip().split("\n\n")

                for chunk in text_chunks:
                    if chunk[0] == "\n":
                        chunk = chunk[1:]
                    new_chunk = (
                        file_name.strip().split(".")[0] + "-" + str(id),
                        chunk,
                        file_name,
                    )
                    chunks.append(new_chunk)
        text_chunks = [chunk for id, chunk, filename in chunks]
        vector_chunk_list = self.embedder.embed(text_chunks)
        joined_chunks = self.__create_combined_list(chunks, vector_chunk_list)
        print(joined_chunks)
        return joined_chunks

    def retrieve(self, query, top_k=TOP_K, threshold=MIN_SCORE):
        """Return transparent lore evidence ranked by meaning."""
        return []

    def build_grounded_prompt(self, stats, evidence):
        """Combine live pet stats with retrieved lore."""
        return ""

    def reply(self, user_message, stats):
        """Return one grounded reply for the Flask talk route."""
        return FALLBACK

    def reset_history(self):
        self.history.clear()
