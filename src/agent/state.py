"""Agent State Definition.

Tracks everything the agent needs across steps in the LangGraph workflow.
"""

from typing import Annotated
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed between all nodes in the agent graph."""

    # User input
    question: str

    # Conversation
    messages: Annotated[list[BaseMessage], add]

    # Routing
    route: str  # "vectorstore" | "web_search" | "direct" | "compare" | "literature_review"

    # Retrieval
    documents: list[dict]  # Retrieved doc chunks with metadata

    # Generation
    generation: str
    citations: list[dict]

    # Quality control
    groundedness_score: float
    hallucination_feedback: str
    retry_count: int

    # Agent trace (for UI streaming)
    trace: Annotated[list[str], add]
