"""
Schedule Management:
Maintain a schedule in a text file (schedule.txt).
Handle user queries related to scheduling through a specialized function.

main_helper/schedule_manager.py: Manages schedule-related queries by interacting
with schedule.txt.
"""

from openai import OpenAI

from datetime import datetime
from main_helper.gpt_response import client


# Function to get today's date in the desired format
def get_today_date():
    return datetime.today().strftime('%Y-%m-%d')


# If the schedule is empty, prompt the user to write something
def scheduler_function(given_input):
    # Read the schedule from the file
    with open("schedule.txt") as f:
        contents = f.readlines()
    if not contents:
        print("Please write a prompt in the box above")
    else:
        # Get user input
        user_prompt = input("Please enter your prompt: ")

        # Format the input prompt
        today_date = get_today_date()
        formatted_prompt = f"q: today is {today_date}, {user_prompt}\nr: \n"

        # Append the formatted prompt to the schedule content
        contents.append(formatted_prompt)

        # Join the contents to create the prompt for the AI
        combined_prompts = "".join(contents)

        # Initialize the OpenAI client
        # client = OpenAI()

        # Get the AI completion
        completion = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system",
                 "content": "You are an AI assistant, really great at helping manage schedules. Follow the given pattern given by the user. Respond then update schedule as done before."},
                {"role": "user", "content": combined_prompts}
            ]
        )

        # Extract the AI response
        txt_response = completion.choices[0].message.content.strip()

        # Prepare the final formatted output to append to the file
        final_output = f"q: {user_prompt}\nr: {txt_response}\n"

        # Append the AI response and the final formatted output to the schedule file
        with open("schedule.txt", "a") as f:
            f.write(final_output + '\n')

        # Read the file again and print only the latest response, stopping if it encounters "date,description"
        with open("schedule.txt") as f:
            lines = f.readlines()
            # Iterate from the end to find the last "r: " response
            for line in reversed(lines):
                if line.startswith("r: "):
                    print(line.strip()[3:])  # Print everything after "r: "
                    return line.strip()[3:]  # Return everything after "r: "
