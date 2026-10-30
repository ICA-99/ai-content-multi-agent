import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app import prompts
from app.config import get_settings
from app.llm import LLMFatalError, generate, generate_structured
from app.progress import log
from app.state import PipelineState

logger = logging.getLogger(__name__)

MAX_TWEET_LEN = 280
MAX_LINKEDIN_LEN = 1300


# ---------------- Output schemas ----------------
# Flat schemas: every field required, no extra properties, no defaults.
# This is the shape constrained decoding handles most reliably.
# Numeric ranges and text lengths are enforced in code, not in the schema.

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchQueries(StrictModel):
    queries: list[str] = Field(description="Exactly 3 distinct web search queries")


class Critique(StrictModel):
    clarity: float = Field(description="Score from 0 to 10")
    accuracy: float = Field(description="Score from 0 to 10")
    structure: float = Field(description="Score from 0 to 10")
    tone: float = Field(description="Score from 0 to 10")
    issues: list[str] = Field(
        description=(
            "Most important problems first. Each item has the form "
            "'TYPE: what is wrong. Fix: what should change.' where TYPE is one of "
            "FACTUAL, CLARITY, STRUCTURE or TONE. Empty list if there are no issues."
        )
    )


class Thread(StrictModel):
    tweets: list[str] = Field(
        description=(
            "6 to 8 tweets in order, each starting with its number such as '1/', "
            "each 270 characters or fewer"
        )
    )


# ---------------- Small helpers ----------------

def _clamp(x: float) -> float:
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _brief(state: PipelineState) -> str:
    return (
        f"Topic: {state['topic']}\n"
        f"Audience: {state.get('audience', 'general readers')}\n"
        f"Tone: {state.get('tone', 'clear and friendly')}"
    )


def _blog_body(state: PipelineState) -> str:
    """The final blog without the appended Sources list (saves tokens for social posts)."""
    return state["final_blog"].split("\n\n## Sources")[0].strip()


def _title(markdown: str, default: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", markdown, re.M)
    return match.group(1) if match else default


def _headings(markdown: str) -> list[str]:
    return [h.strip() for h in re.findall(r"^##\s+(.+?)\s*$", markdown, re.M)]


def _fallback_outline(state: PipelineState) -> str:
    return (
        f"# {state['topic']}\n\n"
        "## Introduction\n- The reader's problem and why the topic matters\n\n"
        "## Core Concepts\n- Key definitions and ideas explained simply\n\n"
        "## How It Works in Practice\n- Practical steps and examples\n\n"
        "## Common Mistakes and Limitations\n- Pitfalls and how to avoid them\n\n"
        "## Conclusion\n- Key takeaway and a call to action\n"
    )


def _fallback_linkedin(state: PipelineState) -> str:
    blog = _blog_body(state)
    parts = [_title(blog, state["topic"])]
    topics = _headings(blog)[:4]
    if topics:
        parts.append("What the article covers:\n" + "\n".join(f"• {t}" for t in topics))
    parts.append("Which of these matters most in your own work?")
    return "\n\n".join(parts)[:MAX_LINKEDIN_LEN]


def _fallback_thread(state: PipelineState) -> list[str]:
    blog = _blog_body(state)
    topics = _headings(blog)[:6]
    tweets = [f"1/ {_title(blog, state['topic'])}"]
    tweets += [f"{i}/ {t}" for i, t in enumerate(topics, start=2)]
    return tweets


def _shorten_tweet(text: str) -> str:
    """Enforce the tweet limit in code, cutting at a word boundary."""
    if len(text) <= MAX_TWEET_LEN:
        return text
    cut = text[: MAX_TWEET_LEN - 1]
    idx = cut.rfind(" ")
    if idx > MAX_TWEET_LEN * 0.6:
        cut = cut[:idx]
    return cut.rstrip(" ,;:-") + "…"


def _clean_tweets(raw_tweets: list[Any]) -> list[str]:
    tweets = []
    for t in raw_tweets:
        t = str(t).strip()
        if t:
            tweets.append(_shorten_tweet(t))
    return tweets


# ---------------- Researcher ----------------

def researcher(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Researcher**")

    settings = get_settings()
    model = settings.researcher_model
    sources: list[dict[str, Any]] = []
    client = None

    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)
        except Exception:
            logger.exception("Tavily could not be initialised")
            log("⚠️ Tavily could not be initialised. Skipping web research.")
    else:
        log("ℹ️ Web research skipped because Tavily is not configured.")

    if client is not None:
        log("🔎 Generating search queries...")
        try:
            queries = generate_structured(
                SearchQueries,
                prompts.QUERY_SYSTEM,
                _brief(state),
                model,
                max_tokens=settings.max_tokens_queries,
            ).queries[:3]
        except LLMFatalError:
            raise
        except Exception:
            logger.exception("Query generation failed; using the topic")
            queries = []
        queries = [q.strip() for q in queries if q and q.strip()] or [state["topic"]]

        seen: set[str] = set()

        for i, q in enumerate(queries, start=1):
            log(f"🌐 Searching the web ({i}/{len(queries)})...")
            try:
                results = client.search(q, max_results=4).get("results", [])
            except Exception:
                logger.exception("Tavily search failed for %r", q)
                log(f"⚠️ Search {i} failed. Continuing...")
                continue

            for r in results:
                url = r.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "content": (r.get("content") or "")[:1200],
                })

        log(f"📚 Collected {len(sources)} unique sources.")

    if sources:
        log("🧠 Summarizing research findings...")
        blob = "\n\n".join(
            f"[{i + 1}] {s['title']} ({s['url']})\n{s['content']}"
            for i, s in enumerate(sources)
        )
        try:
            notes = generate(
                prompts.RESEARCH_SYSTEM,
                f"{_brief(state)}\n\nSearch results:\n{blob}",
                model,
                max_tokens=settings.max_tokens_research,
            )
        except LLMFatalError:
            raise
        except Exception:
            logger.exception("Research summary failed; using raw snippets")
            log("⚠️ Summarizing failed. Using the raw search snippets as notes.")
            notes = "\n\n".join(
                f"- {s['title']} ({s['url']}): {s['content'][:500]}" for s in sources
            )
    else:
        notes = (
            "No live web research available. Rely only on well-established "
            "knowledge and avoid specific statistics, dates or quotes."
        )

    log("✅ Research completed.")
    return {"research_notes": notes, "sources": sources, "revision": 0}


# ---------------- Planner ----------------

def planner(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Planner**")
    log("🧩 Creating the article outline...")

    settings = get_settings()
    try:
        outline = generate(
            prompts.PLANNER_SYSTEM,
            f"{_brief(state)}\n\nResearch notes:\n{state['research_notes']}",
            settings.planner_model,
            max_tokens=settings.max_tokens_planner,
        )
    except LLMFatalError:
        raise
    except Exception:
        logger.exception("Planner failed; using a generic outline")
        log("⚠️ Planner failed. Using a generic outline.")
        outline = _fallback_outline(state)

    log("✅ Article outline completed.")
    return {"outline": outline}


# ---------------- Writer ----------------

def writer(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Writer**")

    settings = get_settings()
    model = settings.writer_model
    revision = state.get("revision", 0) + 1
    critique = state.get("critique")

    if critique and state.get("draft"):
        log(f"✏️ Revising the draft (round {revision})...")
        try:
            draft = generate(
                prompts.REVISE_SYSTEM,
                f"{_brief(state)}\n\n"
                f"Current draft:\n{state['draft']}\n\n"
                f"Critic feedback:\n{critique['feedback']}\n\n"
                f"Research notes:\n{state['research_notes']}",
                model,
                max_tokens=settings.max_tokens_writer,
            )
        except LLMFatalError:
            raise
        except Exception:
            logger.exception("Revision failed; keeping the previous draft")
            log("⚠️ Revision failed. Keeping the previous draft.")
            draft = state["draft"]
    else:
        log("✍️ Creating the first draft...")
        draft = generate(
            prompts.WRITER_SYSTEM,
            f"{_brief(state)}\n\n"
            f"Outline:\n{state['outline']}\n\n"
            f"Research notes:\n{state['research_notes']}",
            model,
            max_tokens=settings.max_tokens_writer,
        )

    log(f"✅ Draft completed (round {revision}).")
    return {"draft": draft, "revision": revision}


# ---------------- Critic ----------------

def critic(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Critic**")
    log("🔍 Evaluating draft quality...")

    settings = get_settings()

    try:
        c = generate_structured(
            Critique,
            prompts.CRITIC_SYSTEM,
            (
                f"{_brief(state)}\n\n"
                f"Draft:\n{state['draft']}\n\n"
                f"Research notes:\n{state['research_notes']}"
            ),
            settings.critic_model,
            max_tokens=settings.max_tokens_critic,
        )

        scores = [
            _clamp(c.clarity),
            _clamp(c.accuracy),
            _clamp(c.structure),
            _clamp(c.tone),
        ]

        issues = [i.strip() for i in c.issues if i and i.strip()]
        feedback = "\n\n".join(issues) or "Improve clarity, accuracy, structure and tone."

    except LLMFatalError:
        raise
    except Exception:
        # The critic is unavailable: accept the draft instead of burning more tokens
        # on revision rounds that have no real feedback.
        logger.exception("Critic failed; accepting the draft")
        log("⚠️ Critic is unavailable. Accepting the current draft.")
        scores = [settings.quality_threshold] * 4
        feedback = "Improve clarity, accuracy, structure and tone."

    avg = round(sum(scores) / len(scores), 1)
    log(f"📊 Quality evaluation completed: {avg}/10.")

    return {
        "critique": {
            "clarity": scores[0],
            "accuracy": scores[1],
            "structure": scores[2],
            "tone": scores[3],
            "feedback": feedback,
        },
        "score": avg,
    }


# ---------------- Editor ----------------

def editor(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Editor**")
    log("🖊️ Polishing the draft...")

    settings = get_settings()
    try:
        edited = generate(
            prompts.EDITOR_SYSTEM,
            f"Draft:\n{state['draft']}\n\nResearch notes:\n{state['research_notes']}",
            settings.editor_model,
            max_tokens=settings.max_tokens_editor,
        )
    except LLMFatalError:
        raise
    except Exception:
        logger.exception("Editor failed; using the unpolished draft")
        log("⚠️ Editor failed. Using the unpolished draft.")
        edited = state["draft"]

    sources = state.get("sources", [])
    if sources:
        log(f"📎 Adding {len(sources)} source links...")
        refs = "\n".join(
            f"- [{s['title'] or s['url']}]({s['url']})" for s in sources
        )
        edited = f"{edited}\n\n## Sources\n{refs}"

    log("✅ Final blog post completed.")
    return {"final_blog": edited}


# ---------------- LinkedIn ----------------

def _enforce_linkedin_limit(post: str, model: str) -> str:
    """Models count characters poorly, so check the limit in code."""
    if len(post) <= MAX_LINKEDIN_LEN:
        return post

    log("✂️ LinkedIn post is over the character limit. Shortening...")
    try:
        post = generate(
            prompts.LINKEDIN_SYSTEM,
            f"Shorten this LinkedIn post to under {MAX_LINKEDIN_LEN - 150} characters. "
            "Keep the hook, the takeaways, the closing question and the hashtags. "
            f"Return only the post.\n\n{post}",
            model,
            max_tokens=get_settings().max_tokens_linkedin,
        )
    except LLMFatalError:
        raise
    except Exception:
        logger.exception("Shortening the LinkedIn post failed")

    if len(post) <= MAX_LINKEDIN_LEN:
        return post

    cut = post[:MAX_LINKEDIN_LEN]
    boundary = max(cut.rfind(". "), cut.rfind("\n"))
    return cut[: boundary + 1].rstrip() if boundary > MAX_LINKEDIN_LEN * 0.5 else cut.rstrip()


def linkedin(state: PipelineState) -> dict[str, Any]:
    log("🔵 **LinkedIn**")
    log("💼 Generating the LinkedIn post...")

    settings = get_settings()
    model = settings.social_model

    try:
        post = generate(
            prompts.LINKEDIN_SYSTEM,
            f"Blog post:\n{_blog_body(state)}",
            model,
            max_tokens=settings.max_tokens_linkedin,
        )
        post = _enforce_linkedin_limit(post, model)
    except LLMFatalError:
        raise
    except Exception:
        logger.exception("LinkedIn generation failed; using a simple fallback post")
        log("⚠️ LinkedIn generation failed. Using a simple fallback post.")
        post = _fallback_linkedin(state)

    log("✅ LinkedIn post completed.")
    return {"linkedin_post": post}


# ---------------- Twitter / X ----------------

def twitter(state: PipelineState) -> dict[str, Any]:
    log("🔵 **Twitter/X**")
    log("🐦 Generating the Twitter/X thread...")

    settings = get_settings()
    model = settings.social_model
    user = f"Blog post:\n{_blog_body(state)}"
    raw_tweets: list[Any] = []

    try:
        raw_tweets = generate_structured(
            Thread,
            prompts.THREAD_SYSTEM,
            user,
            model,
            max_tokens=settings.max_tokens_twitter,
        ).tweets
    except LLMFatalError:
        raise
    except Exception:
        # Fallback: ask for plain text and split on blank lines.
        logger.exception("Structured thread failed; using plain-text fallback")
        try:
            text = generate(
                prompts.THREAD_SYSTEM
                + "\nIgnore the instruction about structured output. Write plain text: "
                "each tweet in its own paragraph, separated by a blank line.",
                user,
                model,
                max_tokens=settings.max_tokens_twitter,
            )
            raw_tweets = text.split("\n\n")
        except LLMFatalError:
            raise
        except Exception:
            logger.exception("Plain-text thread failed too")

    tweets = _clean_tweets(raw_tweets)
    if not tweets:
        log("⚠️ Thread generation failed. Using a simple fallback thread.")
        tweets = _clean_tweets(_fallback_thread(state))

    log(f"✅ Twitter/X thread completed: {len(tweets)} tweets.")
    return {"twitter_thread": tweets}


# ---------------- Routing ----------------

def route_after_critic(state: PipelineState) -> str:
    settings = get_settings()

    if state.get("score", 0) >= settings.quality_threshold:
        log("🎯 Quality threshold reached. Moving to Editor.")
        return "editor"

    if state.get("revision", 0) >= settings.max_revisions:
        log("🛑 Maximum revision rounds reached. Moving to Editor.")
        return "editor"

    log("🔁 Quality threshold not reached. Returning to Writer.")
    return "writer"