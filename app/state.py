from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    # inputs
    topic: str
    audience: str
    tone: str

    # research + planning
    research_notes: str
    sources: list[dict[str, Any]]
    outline: str

    # writing loop
    draft: str
    critique: dict[str, Any]
    score: float
    revision: int

    # outputs
    final_blog: str
    linkedin_post: str
    twitter_thread: list[str]