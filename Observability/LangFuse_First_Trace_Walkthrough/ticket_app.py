"""
The starter app routes a support ticket to a team.
The supplied app logic lives in this file

What the app does
The app lets the user choose one prepared support ticket:

1. Heating failure
2. Double charge

It then sends the selected ticket through a small LangChain pipeline.

Inside this file, the app uses:

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
For this walkthrough, LangChain has one small job: connect the prompt to the model so LangFuse can trace the run cleanly.

ticket_app.py runs the app.
build_config() tells LangChain whether to send trace data to LangFuse.

LangChain can send trace data to LangFuse through a callback handler.
The supplied app already has this pipeline:

ticket -> prompt template -> model call
"""


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, AuthenticationError, RateLimitError


load_dotenv()

MODEL = "gpt-4o-mini"


EXAMPLES = {
    "1": {
        "label": "Heating failure",
        "ticket": "My heating has stopped working and the apartment is cold.",
        "ticket_type": "maintenance",
    },
    "2": {
        "label": "Double charge",
        "ticket": "I was charged twice for this month's rent.",
        "ticket_type": "billing",
    },
}


def build_chain():
    """Create the small LangChain prompt to model pipeline."""
    prompt = ChatPromptTemplate.from_messages([ # ChatPromptTemplate prepares the messages sent to the model;
        (
            "system",
            "You route support tickets to the correct team. "
            "Choose one team: maintenance, billing, account-access, or general-support. "
            "Return exactly two short lines: TEAM: ... and REASON: ...",
        ),
        ("user", "Ticket: {ticket}"),
    ])

    model = ChatOpenAI(model=MODEL) # ChatOpenAI calls the model;
    return prompt | model # creates a chain where the prompt output goes into the model.


def route_ticket(ticket, ticket_type, build_config):
    """Route one support ticket with optional tracing config."""
    chain = build_chain()
    config = build_config(ticket_type)

    if config is None:
        config = {}

    # The following piece of code calls the config function when it runs the chain
    # Here is where the LangFuse callback gets attached to the model call.
    result = chain.invoke(
        {"ticket": ticket},
        config=config,
    )

    return result.content


def choose_ticket():
    """Ask the learner which prepared ticket to run."""
    print("Choose a support ticket:")
    print("1. Heating failure")
    print("2. Double charge")

    choice = input("Enter 1 or 2: ").strip()
    return choice, EXAMPLES.get(choice)


def run_app(build_config):
    """Run the walkthrough app with the supplied config builder.
    run_app
    input: build_config contains the link to LangFuse instance, with the project name '"LangFuse Walkthrough Ticket"'
    The link is pre-configured with the public key's from the .env file.
    """
    choice, example = choose_ticket()

    if example is None:
        print("\nINVALID CHOICE")
        print("--------------")
        print(f"'{choice}' is not one of the available options.")
        print("Choose 1 or 2.")
        return

    try:
        response = route_ticket(
            example["ticket"],
            example["ticket_type"],
            build_config,
        )
    except AuthenticationError:
        print("\nMODEL AUTHENTICATION FAILED")
        print("---------------------------")
        print("The model client could not authenticate.")
        return
    except RateLimitError:
        print("\nMODEL RATE LIMIT HIT")
        print("--------------------")
        print("The model service rejected the request because of rate limits.")
        return
    except APIConnectionError:
        print("\nMODEL CONNECTION FAILED")
        print("-----------------------")
        print("The app could not connect to the model service.")
        return

    print("\nMODEL ROUTING RESULT")
    print("--------------------")
    print(response)