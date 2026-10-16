DigiPet exercise
- Uses RAG/embedding via OPENAI
- Uses Flask and SQLite
- Uses multiple file read 
- A nice list comprehension to create a list of the 1st tuple item in a list of tuples
- an interesting use for providing a fallback object: def __init__(self, embedding=None): self.embedding = embedding or OpenAIEmbedding()
- 