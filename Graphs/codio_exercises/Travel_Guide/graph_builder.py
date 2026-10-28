"""LangGraph topology for the travel agent team."""

from langgraph.graph import END, START, StateGraph

from coordinator import coordinator_node
from specialist_agents import hotel_node, transport_node, weather_node
from team_state import TravelState


def build_graph() -> object:
    """Compile the parallel specialist team and coordinator join."""
    print("call build_graph()")
    graph = StateGraph(TravelState)


    # join all branches at coordinator, connect coordinator to END, compile,
    # and return the graph.
    graph = StateGraph(TravelState)
    graph.add_node("transport", transport_node)
    graph.add_node("weather", weather_node)
    graph.add_node("hotel", hotel_node)
    graph.add_node("coordinator", coordinator_node)
    for node in ("transport", "weather", "hotel"):
        graph.add_edge(START, node)
    graph.add_edge(
        ["transport", "weather", "hotel"],
        "coordinator",
    )
    graph.add_edge("coordinator", END)
    return graph.compile()
