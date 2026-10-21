from flask import Flask, render_template, request
import os
from openai import OpenAI
from prompt_builder import build_prompt
app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)
app = Flask(__name__)
MAX_OUTPUT_LENGTH=4200

def generate_study_help(topic: str, mode: str) -> str:
    prompt = build_prompt(topic, mode)
    system_prompt = "You are a python expert, helping students to program better."
    response = client.responses.create(
        model="gpt-4o-mini",
        messages=[{"role": "system",
                   "content": system_prompt},
                  {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=MAX_OUTPUT_LENGTH,
    )
    # return (
    #     "Starter mode is still active.\n\n"
    #     f"Topic: {topic}\n"
    #     f"Mode: {mode}\n\n"
    #     "Your next job is to connect the OpenAI client and replace this placeholder "
    #     "with real study help."
    # )
    return(response.choices[0].message.content)

@app.route("/", methods=["GET", "POST"])
def index():
    topic = ""
    mode = "quiz"
    result = ""
    error = ""

    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        mode = (request.form.get("mode") or "quiz").strip()

        if not topic:
            error = "Please enter a topic before asking the study assistant for help."
        else:
            result = generate_study_help(topic, mode)

    return render_template(
        "index.html",
        topic=topic,
        mode=mode,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
