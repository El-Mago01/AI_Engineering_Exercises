This code came from a demo from Slawa on 29.10 as part of the Multiple Agents trainings session.
The following learnings can be observed from this demo:
- Working with a team of multiple langchain agents with specialists roles and coordinator role using LangGraph
- Using a TypeDict for LandGraph State
- Working with a standard pydantic report
- Working with a pydantic BaseModel 
- Print ASCII representation of the langgraph or graph using graph.get_graph().print_ascii()
- Use of langchain agents together with langgraph
- Using reports: Annotated[list[dict], operator.add] to ensure new reports are added, not overwritten