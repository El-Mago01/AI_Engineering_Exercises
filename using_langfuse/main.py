import os
import random
import uuid
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI
from output import nicer_print

# Load variables from .env file
load_dotenv()

# Create OpenAI client instance
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generic catalogue of dishes the kitchen can cook
ALL_DISHES = [
    "Bolognese",
    "Shawarma",
    "Doner Kebab",
    "Sushi",
    "Margherita Pizza",
    "Pad Thai",
    "Chicken Curry",
    "Caesar Salad",
    "Beef Tacos",
    "Ramen",
]

# The trick: only expose 5 random dishes to the LLM on each call,
# so the available "menu" changes from run to run.
available_dishes = random.sample(ALL_DISHES, 5)


def _generate_user_input() -> str:
    generate_user_input_response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You generate a single, realistic meal request as if written by "
            "a hungry person deciding what to eat. Vary the mood, cravings, "
            "and phrasing each time. "
            "Return only the request, with no extra commentary."
        ),
        input="Generate one random meal request.",
    )

    return generate_user_input_response.output_text.strip()


langfuse = get_client()


# Attach a session id so multiple runs (each its own trace) are grouped
# into one replayable session thread in Langfuse. A constant user_id is
# added so traces can be attributed to (and filtered by) a specific user.
with propagate_attributes(session_id=str(uuid.uuid4()), user_id="demo-user"):
    # Group both OpenAI calls under a single parent trace so the whole
    # request cycle is observable as one unit in Langfuse.
    with langfuse.start_as_current_observation(name="waiter-bot testing"):
        user_input = _generate_user_input()
        # user_input = "I want a big bowl of creamy mac and cheese with a crispy topping"

        # Fetch the managed prompt from Langfuse and fill its {{dishes}}
        # placeholder with this run's available dishes.
        # meal_prompt = langfuse.get_prompt("meal-prep-prompt")
        meal_prompt = f"""You are a waiter in a restaurant and you are given a list of available meals the kitchen can cook.
        Please recommend a best fitting meal for this customer. Listen well to his input and make the recommendation based on 
        what he or she is saying.
        
        Here are the available meals:
        {", ".join(available_dishes)}
        """
        # instructions = meal_prompt.compile(dishes=", ".join(available_dishes))
        instructions = meal_prompt

        user_input = _generate_user_input()
        response = client.responses.create(  # type: ignore[call-overload]
            model="gpt-4o-mini",
            instructions=instructions,
            input=user_input,
        )

nicer_print(user_input, available_dishes, response.output_text)
