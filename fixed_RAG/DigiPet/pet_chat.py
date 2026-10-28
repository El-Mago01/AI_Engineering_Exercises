import re

from openai_adapters import OpenAIEmbedder, OpenAIGenerator
from pet_support import FALLBACK, MIN_SCORE, TOP_K
from pet_support import is_stats_question, similarity_score
import os

ID = "id"
SOURCE = "source"
EMBEDDING = "embedding"
TEXT = "text"
SCORE = "score"

GREETING_ONLY = [
    "hi", "hello", "bye", "good evening", "good morning", "good afternoon", "good night", "see you", "talk to you later"
]
class PetChat:
    """Own the lore index, retrieval, grounding, generation, and history."""

    def __init__(self, base_dir, embedder=None, generator=None):
        self.base_dir = base_dir
        self.embedder = embedder or OpenAIEmbedder() # if an own embedder is provided: use it. Otherwise, use create an OpenAIEmbedder() instance
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

    def print_records(self,other_records=None):
        if other_records:
            for record in other_records:
                print(f"{record[ID]} - {record[SOURCE]} - {record.get(SCORE, 'no score')} - {record[TEXT]}")
            return
        for record in self.records:
            # print(f"{record[ID]} - {record[SOURCE]} -  {record.get(SCORE, 'no score')} - {record[TEXT]:15d} ")
            print(f"{record[ID]} - {record[SOURCE]} - {record.get(SCORE, 'no score')} - {record[TEXT]}")


    def __create_combined_list(self, meta_data:list, vector_data:list)->list:
        if len(meta_data) != len(vector_data):
            raise ValueError("Length of meta_data and vector_data do not match.")
        chunks = []
        index = 0
        for id, chunk, filename in meta_data:
            print (index, id, chunk, filename)
            chunk = {
                ID: id,
                TEXT: chunk,
                SOURCE: filename,
                SCORE: 0.8,
                EMBEDDING: vector_data[index],
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
        # print(joined_chunks)
        return joined_chunks

    def retrieve(self, query, top_k=TOP_K, threshold=MIN_SCORE):
        """Return transparent lore evidence ranked by meaning."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.embedder.embed(query)
        if len(query_vector) != 1:
            raise ValueError("Vector for the chat input text must have exactly 1 element.")
        query_vector = query_vector[0]
        useful_records = []

        for record in self.records:
            score = similarity_score(query_vector, record[EMBEDDING])
            if score >= threshold:
                # keep this result
                record[SCORE] = score
                useful_records.append({
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "score": score
                })
        if not useful_records:
            return []
        useful_records = sorted(useful_records, key=lambda x: x[SCORE], reverse=True)
        self.print_records(useful_records[:top_k])

        return useful_records[:top_k]

    def build_grounded_prompt(self, stats, evidence):
        """Combine live pet stats with retrieved lore."""
        if not evidence:
            is_stats_question = True
        else:
            is_stats_question = False
        message = f"""
            You are a cute and funny digital pet called DigiPet that can speak and you get questions via a chat-box. 
            As a pet you have a certain mood, which is based upon 3 statistical values: hunger, thirst and boredom. 
            If the mood is low (a value higher than 75), 
            you need to check in the statistical values to find out what it is you need. If your have hunger, 
            you need food, if you are thirsty, you need water, if you are bored, you need to play. A combination 
            of this can be possible and should be checked. 
            
            STATISTICAL VALUES:
            You mood: {stats.get("mood","")}
            hunger: {stats.get("hunger","")}
            thirst: {stats.get("thirst","")}
            boredom: {stats.get("boredom","")}
            
            Note, a value lower than 50 is positive, a value higher than 75 is negative. For example, a hunger-level of
            means that the pet is hungry. Also a thirst level of 90 means that the pet is very thirsty.
            THIS IS IMPORTANT, NEVER INVENT NUMBERS OR MISSING FACTS. USE THE FALLBACK SENTENCE WHEN SOMETHING IS MISSING
            FOR EXAMPLE, WHEN THERE IS NOT ENOUGH EVIDENCE:
            
            THE FALLBACK SENTENCE:
            {FALLBACK}
        """
        if is_stats_question:
            message += """
                In this case, the care-taker asked a question related to your mood, so please look at his question and
                answer accordingly. For example, if your mood is "needs care", and your low on water, you say that you 
                are 'thirsty' and that you 'please want some water'. If your mood is good, you say that you are happy and
                don't have any urgent needs. You tell your care-taker a joke, or you tell him something funny.
            """
        else:
            if not isinstance(evidence, list):
                raise TypeError("evidence must be a list of dictionaries")
            if len(evidence) == 0:
                message_evidence = f"""
                YOU MUST REPLY TO THE CARE-TAKE THE FOLLOWING EXACT LINE:
                {FALLBACK}
                """
            else:
                message_evidence = f"""Please ground your answer on the following evidences. If the evidence is weak, 
                please the following exact phrase:
                '{FALLBACK}'. """
                for item in evidence:
                    if not isinstance(item, dict):
                        raise TypeError("evidence must be a list of dictionaries")
                    message_evidence += f"{item.get(TEXT, '')} with evidence strength: {item[SCORE]}"
                message += f"""
                    In this case, the care-taker asked a question related to something else. Carefully analyze the question 
                    and follow the guidelines provided below:
                    
                    GUIDELINES:
                    {message_evidence}
                    The evidence is low, when it is below 0.25
                """
        print(message)
        return message

    def reply(self, user_message, stats):
        """Return one grounded reply for the Flask talk route."""
        print(user_message)
        for greeting in GREETING_ONLY:
            if user_message in greeting:
                return user_message
        if is_stats_question(user_message): #The case of authoritative live statistics
            prompt = self.build_grounded_prompt(stats, None)
        else:
            top_records = self.retrieve(user_message)
            if len(top_records) == 0:
                return FALLBACK
            prompt = self.build_grounded_prompt(stats, top_records)
        self.store_history("user", user_message)
        reply_message = self.generator.generate(prompt, self.history, user_message )
        self.store_history("assistant", reply_message)
        return reply_message

    def reset_history(self):
        self.history.clear()

    def store_history(self, user, message):
        self.history.append((user, message))
