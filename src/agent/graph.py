"""LangGraph Workflow Assembly.

Builds the full agent graph with conditional routing, self-correction loops,
and retry limits. This is the core orchestration layer.

Flow:
    START → route_query → [retrieve | web_search | direct_answer]
                              ↓           ↓
                          grade_documents ←┘
                              ↓
                     [has docs?] → rewrite_query → route_query (retry)
                              ↓
                       generate_answer
                              ↓
                    check_hallucination
                              ↓
                 [grounded?] → generate_answer (retry with feedback)
                              ↓
                             END
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    route_query,
    retrieve_from_vectorstore,
    web_search,
    grade_documents,
    generate_answer,
    direct_answer,
    check_hallucination,
    rewrite_query,
    should_route_to,
    should_retry_or_finish,
    has_relevant_docs,
)


def build_graph() -> StateGraph:
    """Construct and compile the agent workflow graph."""
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve_from_vectorstore)
    graph.add_node("web_search", web_search)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite_query", rewrite_query)

    # --- Entry point ---
    graph.set_entry_point("route_query")

    # --- Conditional routing from router ---
    graph.add_conditional_edges(
        "route_query",
        should_route_to,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "direct_answer": "direct_answer",
        },
    )

    # --- After retrieval / web search → grade ---
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("web_search", "grade_documents")

    # --- After grading → generate or rewrite ---
    graph.add_conditional_edges(
        "grade_documents",
        has_relevant_docs,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
        },
    )

    # --- Rewrite loops back to routing ---
    graph.add_edge("rewrite_query", "route_query")

    # --- After generation → hallucination check ---
    graph.add_edge("generate_answer", "check_hallucination")

    # --- After hallucination check → finish or retry ---
    graph.add_conditional_edges(
        "check_hallucination",
        should_retry_or_finish,
        {
            "finish": END,
            "retry": "generate_answer",
        },
    )

    # --- Direct answer goes straight to END ---
    graph.add_edge("direct_answer", END)

    return graph.compile()


# Singleton compiled graph
agent = build_graph()


def run_agent(question: str) -> dict:
    """Run the agent on a question and return the final state.

    Returns:
        Dict with: generation, citations, groundedness_score, documents, trace, route
    """
    initial_state: AgentState = {
        "question": question,
        "messages": [],
        "route": "",
        "documents": [],
        "generation": "",
        "citations": [],
        "groundedness_score": 0.0,
        "hallucination_feedback": "",
        "retry_count": 0,
        "trace": [],
    }

    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state.get("generation", ""),
        "citations": final_state.get("citations", []),
        "groundedness_score": final_state.get("groundedness_score", 0.0),
        "documents": final_state.get("documents", []),
        "trace": final_state.get("trace", []),
        "route": final_state.get("route", ""),
    }


def stream_agent(question: str):
    """Stream agent execution, yielding state updates at each node.

    Yields:
        Tuples of (node_name, partial_state_update)
    """
    initial_state: AgentState = {
        "question": question,
        "messages": [],
        "route": "",
        "documents": [],
        "generation": "",
        "citations": [],
        "groundedness_score": 0.0,
        "hallucination_feedback": "",
        "retry_count": 0,
        "trace": [],
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in event.items():
            yield node_name, state_update
