import streamlit as st
from langgraph.errors import GraphRecursionError

from app.config import get_settings
from app.graph import get_graph
from app.llm import LLMFatalError, LLMUnavailableError
from app.nodes import MAX_LINKEDIN_LEN, MAX_TWEET_LEN

st.set_page_config(page_title="Content Pipeline Agent", page_icon="✍️", layout="wide")


def run_pipeline(topic: str, audience: str, tone: str) -> dict:
    """Run the graph and show live progress."""
    initial = {"topic": topic, "audience": audience, "tone": tone}
    final: dict = {}

    with st.status("🤖 Agents are working...", expanded=True) as status:
        try:
            for mode, chunk in get_graph().stream(
                initial,
                config={"recursion_limit": 60},
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    message = chunk.get("message")
                    if message:
                        st.write(message)

                elif mode == "updates":
                    for node, update in chunk.items():
                        if isinstance(update, dict):
                            final.update(update)
                        st.write(f"✔️ {node.replace('_', ' ').title()} completed.")
        except Exception:
            # Show a red "failed" state in the status box, then let main() explain why.
            status.update(label="❌ Run failed", state="error", expanded=True)
            raise

        status.update(
            label="✅ Content generation completed!",
            state="complete",
            expanded=False,
        )

    return {
        "topic": topic,
        "final_blog": final.get("final_blog", ""),
        "linkedin_post": final.get("linkedin_post", ""),
        "twitter_thread": final.get("twitter_thread", []),
        "score": final.get("score"),
        "revisions": final.get("revision"),
        "critique": final.get("critique"),
        "sources": final.get("sources", []),
    }


def explain_error(exc: Exception) -> str:
    """Turn an exception into a short, readable message."""
    if isinstance(exc, LLMFatalError):
        return str(exc)

    if isinstance(exc, LLMUnavailableError):
        return (
            f"{exc}\n\n"
            "What you can do: wait a few minutes and try again, switch the model names "
            "in your `.env` file (for example `WRITER_MODEL`), or upgrade your Groq plan."
        )

    if isinstance(exc, GraphRecursionError):
        return (
            "The pipeline looped more times than allowed. "
            "Try lowering `MAX_REVISIONS` in your `.env` file."
        )

    return f"Unexpected error ({type(exc).__name__}): {exc}"


def render_result(res: dict) -> None:
    st.subheader(res["topic"])

    score = res.get("score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final quality score", f"{score}/10" if score is not None else "–")
    c2.metric("Draft rounds", res.get("revisions") or "–")
    c3.metric("Web sources", len(res.get("sources", [])))

    if not res.get("final_blog"):
        st.warning("The pipeline finished without producing a blog post. Please try again.")

    blog_tab, li_tab, tw_tab, qa_tab = st.tabs(
        ["📝 Blog post", "💼 LinkedIn", "🐦 Twitter/X thread", "🔍 Quality & sources"]
    )

    with blog_tab:
        blog = res.get("final_blog", "")
        st.download_button(
            "Download .md", blog, file_name="blog_post.md", mime="text/markdown"
        )
        st.markdown(blog)

    with li_tab:
        post = res.get("linkedin_post", "")
        if len(post) > MAX_LINKEDIN_LEN:
            st.caption(f"⚠️ {len(post)} characters (over the {MAX_LINKEDIN_LEN} limit)")
        else:
            st.caption(f"{len(post)}/{MAX_LINKEDIN_LEN} characters")
        st.text_area("LinkedIn post", post, height=380, label_visibility="collapsed")
        st.download_button("Download .txt", post, file_name="linkedin_post.txt")

    with tw_tab:
        tweets = res.get("twitter_thread", [])
        for i, t in enumerate(tweets, 1):
            with st.container(border=True):
                st.write(t)
                st.caption(f"Tweet {i}/{len(tweets)} · {len(t)}/{MAX_TWEET_LEN}")
        st.download_button(
            "Download .txt", "\n\n".join(tweets), file_name="twitter_thread.txt"
        )

    with qa_tab:
        crit = res.get("critique") or {}

        if crit:
            cols = st.columns(4)
            for col, key in zip(cols, ["clarity", "accuracy", "structure", "tone"]):
                col.metric(key.title(), crit.get(key, "–"))

            st.markdown("**Last critic feedback**")
            st.write(crit.get("feedback", ""))

        st.markdown("**Sources**")
        sources = res.get("sources", [])

        for s in sources:
            st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")

        if not sources:
            st.caption("No web sources. Set TAVILY_API_KEY to enable live research.")


def main() -> None:
    st.title("✍️ Content Pipeline Agent")
    st.caption(
        "One topic in → blog post, LinkedIn post and Twitter/X thread out. "
        "Powered by LangGraph."
    )

    with st.form("new_run"):
        topic = st.text_input("Topic", placeholder="e.g. MLOps for beginners")

        c1, c2 = st.columns(2)
        audience = c1.text_input("Audience", value="curious beginners")
        tone = c2.text_input("Tone", value="clear, friendly and practical")

        submitted = st.form_submit_button("Generate content", type="primary")

    if submitted:
        if len(topic.strip()) < 3:
            st.warning("Please enter a topic (at least 3 characters).")

        elif not get_settings().groq_api_key:
            st.error("GROQ_API_KEY is missing. Add it to your .env file and restart.")

        else:
            try:
                # Keep the result in session_state so it survives reruns.
                st.session_state.result = run_pipeline(
                    topic.strip(),
                    audience.strip() or "general readers",
                    tone.strip() or "clear",
                )
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Run failed: {explain_error(exc)}")

    if st.session_state.get("result"):
        render_result(st.session_state.result)


main()