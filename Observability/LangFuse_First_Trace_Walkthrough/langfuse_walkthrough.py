"""
In Week 1, you looked at why AI applications need observability.
In this walkthrough, you will connect a small LangChain app to LangFuse so you can
inspect what happened during a real model call.

ticket_app.py contains the supplied support-ticket app. It handles the
LangChain prompt, model call, menu, and error handling.
langfuse_walkthrough.py is the file you will edit. Your job is to add
the LangFuse tracing config.

What is LangChain doing here?
LangChain is a framework for connecting prompts, models, tools, and callbacks.
In this walkthrough, LangChain has one small job: it connects a prompt template to a model call so LangFuse can trace the run cleanly.
You do not need deep LangChain knowledge for this lesson. Read the supplied app as a simple pipeline:
prepare messages -> call model -> return result

user chooses ticket -> LangChain app -> model call -> LangFuse trace -> dashboard inspection
ChatPromptTemplate creates the messages sent to the model.
ChatOpenAI calls the OpenAI chat model.

Key ideas
Term	        Beginner meaning
Session	        A group of related user interactions.
Trace	        One full run of an AI request or workflow.
Observation	    A recorded item inside a trace.
Generation	    A model call captured inside the trace.
Metadata	    Searchable facts attached to a run.
Tags	        Short labels used to group and filter traces.

You will:
create a LangFuse project;
add LangFuse keys to a .env file;
complete the tracing config in langfuse_walkthrough.py;
run the support-ticket app;
send the run to LangFuse;
inspect the trace dashboard;
run a second ticket and compare the traces.
"""

from langfuse.langchain import CallbackHandler
# CallbackHandler is the LangFuse hook. LangChain calls it while the chain runs, and the handler sends trace
# data to your LangFuse project.
from ticket_app import run_app


TRACE_NAME = "LangFuse Walkthrough Ticket"

def build_config(ticket_type):
    """Create LangFuse tracing config for one ticket run.
    What this does:
        callbacks tells LangChain to send the run to LangFuse;
        run_name gives the trace a name you can look for;
        metadata stores safe facts about the run;
        tags help you filter related traces in the dashboard.
    """
    langfuse_handler = CallbackHandler()

    return {
        "callbacks": [langfuse_handler],
        "run_name": TRACE_NAME,
        "metadata": {
            "ticket_type": ticket_type,
        },
        "tags": ["langfuse-walkthrough"],
    }


if __name__ == "__main__":
    run_app(build_config)