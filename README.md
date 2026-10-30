# Content Pipeline Agent (simple version)

One topic in, three assets out: a blog post, a LinkedIn post and a Twitter/X thread.
Built with LangGraph and Streamlit. No database, no API server.

## Graph

```
START -> researcher -> planner -> writer -> critic --(score < threshold and rounds left)--> writer
                                               |
                                               +--(good enough or max rounds)--> editor --+--> linkedin --> END
                                                                                          +--> twitter  --> END
```

## Project layout

```
streamlit_app.py     UI: takes input, runs the graph, shows live progress and results
app/
  config.py          settings from .env
  llm.py             OpenAI chat model
  state.py           shared graph state
  prompts.py         all prompts
  nodes.py           the agents (nodes) and the critic router
  graph.py           builds the LangGraph graph
  progress.py        turns node updates into readable progress messages
tests/test_pipeline.py
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY (TAVILY_API_KEY optional)
streamlit run streamlit_app.py
```

Open http://localhost:8501

## Docker

```bash
docker build -t content-agent .
docker run --env-file .env -p 8501:8501 content-agent
```

## Test

```bash
pytest -q
```

Results live in the browser session only. Use the download buttons to keep them.
