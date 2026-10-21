import os
import json
from ollama import Client
MAX_SUMMARY_LENGTH = 2000
MAX_HISTORY_LENGTH = 2  # number of chats in the history, excluding the new chat
MODEL = "gemma4:e2b"
client = Client(host="192.168.1.75:11434")

def get_system_dict() -> dict:
    system_prompt = """
    Role : You are a CLI assistant that answers user questions.
    Constraints: 
    - Always answer the user’s question, even if information is incomplete.
    - If information is missing, clearly say what is missing and lower the confidence level and provide the reason in the json output format.
    - Never include any text outside of a JSON object.

    Output format:
    - Respond with a single JSON object with exactly these fields:
    - answer: string – your main reply to the user
    - confidence: string – one of "low", "medium", "high"
    - confidence_reason: string – a short explanation of why you chose that confidence level
    """
    system_dict = {"role": "system", "content": system_prompt}

    return system_dict


def summarize_history_helper(history: str) -> str:
    response = client.chat(
        model=MODEL,
        messages=[{"role": "system",
                   "content": f"You task is to summarize the provided chat history in max {MAX_SUMMARY_LENGTH} words"},
                  {"role": "user", "content": history}],
        # temperature=0.1,
        # max_tokens=MAX_SUMMARY_LENGTH,
    )
    return response.message.content


def summarize_history(messages) -> list:
    # messages[0] contains a dict with the system prompt
    # messages[1] contains a dict with the summary
    # messages[2..n] contains a dict with the chat history, not summarized

    if (len(messages) - 2) <= MAX_HISTORY_LENGTH:  # -2 as the first 2 elements of messages are reserved
        return messages  # nothing to summarize
    if len(messages[1].get("content", "")) > MAX_SUMMARY_LENGTH:
        messages[1] = {"role": "assistant",
                       "content": summarize_history_helper(messages[1].get("content"))}
    history_chat = ""
    for ind in range(2, len(messages) - MAX_HISTORY_LENGTH + 2):
        if messages[ind].get("role", "") == "user":
            history_chat += "Question: "
        elif messages[ind].get("role", "") == "assistant":
            history_chat += "Answer: "

        history_chat += messages[ind].get("content", "") + "\n"
    for ind in range(2, len(messages) - MAX_HISTORY_LENGTH + 2):
        messages.pop(2)
    messages[1] = {"role": "assistant",
                   "content": summarize_history_helper(history_chat)}
    return messages


def get_advice_from_ai(messages: list, new_question: str) -> str:
    messages = summarize_history(messages)
    messages.append({"role": "user", "content": new_question})

    response = client.chat(
        model=MODEL,
        messages=messages,
        # messages = [{"role": "system", "content": messages[0]["content"]},
        #             {"role": "user", "content": messages[-1].get("content")}],
        # temperature=0.9,
        # max_tokens=2000,
    )
    return response


def store_advice(messages: list, advice) -> None:
    answer_to_decode = advice.message.content
    if answer_to_decode[3:7] == "json":
        answer_to_decode = answer_to_decode[8:]
        print(answer_to_decode)
    the_answer = answer_to_decode[-3:]
    if answer_to_decode[-3:] == "```":
        answer_to_decode = answer_to_decode[:-4]
        print(answer_to_decode)
    try:
        decoded_answer = json.loads(answer_to_decode)
    except json.JSONDecodeError as e:
        print("Invalid JSON returned by OpenAI. See message: ", e)
        print(advice.message.content)
        return
    messages.append({"role": "assistant", "content": decoded_answer.get("answer", "")})
    return


def pretty_print(answer):
    answer_to_decode = answer.message.content
    if answer_to_decode[3:7] == "json":
        answer_to_decode = answer_to_decode[8:]
        print(answer_to_decode)
    if answer_to_decode[-3:] == "```":
        answer_to_decode = answer_to_decode[:-4]
        print(answer_to_decode)
    try:
        decoded_answer = json.loads(answer_to_decode)
    except json.JSONDecodeError as e:
        print("Invalid JSON returned by OpenAI. See message: ", e)
        print(decoded_answer)
        return
    ai_answer = decoded_answer.get("answer", "no answer provided")
    confidence = decoded_answer.get("confidence", "no confidence provided")
    confidence_reason = decoded_answer.get("confidence_reason", "no confidence reason provided")
    evaluation_duration = answer.eval_duration
    finish_reason = answer.done_reason
    print("==>", end="")
    if finish_reason == "stop":
        print("\n")
        print("Answer:                ", ai_answer)
        print("Confidence:            ", confidence)
        print("Confidence background: ", confidence_reason)
    else:
        print("Not able to get a complete answer. Answer was interrupted!")
    print("----------------------------------------")
    print(evaluation_duration)
    print(finish_reason)
    print("----------------------------------------")


def main():
    """

    :return:
    """
    # messages variable holds the recent chat history, history summary and system prompt:
    # messages[0] contains system prompt
    # messages[1] contains chat summary
    # messages[2..n] contains the recent chat history
    messages = []
    system_dict = get_system_dict()
    messages.append(system_dict)
    messages.append({"role": "assistant", "content": ""})
    while True:
        chat_line = input(">>> ")
        if chat_line.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif chat_line.lower() in ["?"]:
            for message in messages:
                print("Role: ", message.get("role", ""))
                print("Content: ", message.get("content", ""))
        else:
            advice = get_advice_from_ai(messages, chat_line)
            store_advice(messages, advice)
            pretty_print(advice)


if __name__ == "__main__":
    main()