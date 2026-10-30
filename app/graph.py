from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app import nodes
from app.state import PipelineState


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("researcher", nodes.researcher)
    g.add_node("planner", nodes.planner)
    g.add_node("writer", nodes.writer)
    g.add_node("critic", nodes.critic)
    g.add_node("editor", nodes.editor)
    g.add_node("linkedin", nodes.linkedin)
    g.add_node("twitter", nodes.twitter)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "planner")
    g.add_edge("planner", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"writer": "writer", "editor": "editor"},
    )

    # parallel fan-out: both repurposers run at the same time
    g.add_edge("editor", "linkedin")
    g.add_edge("editor", "twitter")
    g.add_edge("linkedin", END)
    g.add_edge("twitter", END)

    return g.compile()


@lru_cache
def get_graph():
    return build_graph()