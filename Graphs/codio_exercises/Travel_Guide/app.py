"""Serve the real Travel Planning Agent Team dashboard."""

"""
You are building the workflow behind a professional travel-operations dashboard. A disrupted journey needs several 
kinds of evidence, so one model is not asked to do everything.

                       -> Transport agent + transport tool ---\
START -> travel case  -> Weather agent + weather tool --------> Travel coordinator -> END
                       -> Hotel agent + hotel tool -----------/
                       
The three specialists investigate independently. LangGraph waits for every branch, merges their findings, and runs the coordinator once.  

What Makes This a Real Agent?
A model call generates output. An agent goes further: it receives a goal, chooses an allowed action, observes the result, and continues using that result.
Goal -> model chooses tool -> Python runs tool -> model observes result -> finding
     
Each specialist in this assignment has four parts:
a clear role;
a real gpt-5-mini model call;
one restricted tool;
a result contract for the next node.
               
"""
#TODO: What students complete
# Complete four TODO groups across three focused files:
# specialist_agents.py: request, validate, and execute scoped tools;
# specialist_agents.py: return tool output and build cited findings;
# coordinator.py: build the coordinator’s cited plan;
# graph_builder.py: connect the parallel LangGraph.
# The data tools, typed state, reducers, dashboard and worker wrappers are supplied.

from flask import Flask, render_template, request
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from tools import CASES
from travel_team import AgentProtocolError, run_team


app = Flask(__name__)


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    print("index()")
    selected = request.form.get("case_id", "berlin-conference")
    result = None
    error = None

    if request.method == "POST":
        print(f"POST-REQUEST RECEIVED WITH {selected}")
        try:
            result = run_team(selected)
        except NotImplementedError:
            error = "Complete the next TODO in travel_team.py, then try again."
        except ValueError:
            error = "Choose a valid travel case."
        except AgentProtocolError:
            error = "An agent returned an invalid tool request. Check your validation code."
        except AuthenticationError:
            error = "The Codio OpenAI credentials are unavailable."
        except PermissionDeniedError:
            error = "This Codio project cannot access the selected model."
        except RateLimitError:
            error = "The service is busy or the course quota was reached. Try again shortly."
        except APIConnectionError:
            error = "The model service could not be reached. Try again shortly."
        except InternalServerError:
            error = "The model service is temporarily unavailable. Try again shortly."
        except APIStatusError as exc:
            error = f"The model service returned an error ({exc.status_code}). Try again."
        except Exception:
            error = "The travel plan could not be created. Check your code and try again."

    if selected not in CASES:
        selected = "berlin-conference"
    return render_template(
        "index.html", cases=CASES, selected=selected, case=CASES[selected], result=result, error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
