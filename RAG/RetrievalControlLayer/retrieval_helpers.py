import math


def similarity_score(vec_a, vec_b):
    """Return a ranking score for two embedding vectors."""
    inproduct = sum(a * b for a, b in zip(vec_a, vec_b))
    # this creates dot = a1*b1 + a2*b2 + ...., i.e. the internal product of 2 vectors
    vec_a_length = math.sqrt(sum(value * value for value in vec_a))
    # this creates SQRT( (a1)^2 + (a2)^2 + (a3)^2 ...(an)^2 ), i.e. the length of the vectore
    vec_b_length = math.sqrt(sum(value * value for value in vec_b))
    if vec_a_length == 0 or vec_b_length == 0:
        return 0.0
    return inproduct / (vec_a_length * vec_b_length)
