# You do not need to edit this file.

import math
import re


TOP_K = 3
MIN_SCORE = 0.25
FALLBACK = "I do not have enough DigiPet lore to answer that safely."
STAT_KEYWORDS = {
    "hungry",
    "hunger",
    "thirsty",
    "thirst",
    "water",
    "bored",
    "boredom",
    "mood",
    "feel",
    "feeling",
    "status",
    "stats",
    "need",
    "needs",
    "care",
}


def similarity_score(left, right):
    """Return cosine similarity for two numeric vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_size = math.sqrt(sum(value * value for value in left))
    right_size = math.sqrt(sum(value * value for value in right))
    if left_size == 0 or right_size == 0:
        return 0.0
    return dot / (left_size * right_size)


def is_stats_question(message):
    """Return True when the user asks about live DigiPet care stats."""
    words = set(re.findall(r"[a-z]+", message.lower()))
    return bool(words & STAT_KEYWORDS)
