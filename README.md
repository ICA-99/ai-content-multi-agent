# Content Pipeline Agent

An AI-powered multi-agent content generation system that transforms a single topic into three content assets: a **Blog Post, LinkedIn Post, and Twitter/X Thread**.

Built using **LangGraph, LangChain, Groq, Tavily, and Streamlit**, the application follows a structured multi-agent workflow to research, plan, generate, evaluate, and refine content.

## 🚀 Live Demo

**[Try Content Pipeline Agent](https://ai-content-multi-agent-jhkvvkovpmmmqr7v7lpmgh.streamlit.app/)**

**Note:** Content generation may take some time because the application executes multiple LLM-powered agents sequentially. To manage token consumption, reduce the likelihood of API rate-limit errors, and maintain reliable execution, pauses are incorporated between certain LLM calls. Please allow sufficient time for the complete workflow to finish.

## Workflow Architecture

The application uses LangGraph to orchestrate multiple specialized agents.

```text
START
  |
  v
Researcher
  |
  v
Planner
  |
  v
Writer
  |
  v
Critic
  |
  +---- Score below threshold
  |            |
  |            v
  |          Writer (Revision)
  |            |
  |            +----> Critic
  |
  +---- Score acceptable / Maximum revisions reached
               |
               v
             Editor
               |
        +------+------+
        |             |
        v             v
     LinkedIn       Twitter/X
        |             |
        +------+------+
               |
              END
```

### Agent Responsibilities

| Agent | Responsibility |
|---|---|
| Researcher | Researches the given topic using relevant information sources. |
| Planner | Develops the content structure and writing strategy. |
| Writer | Generates the initial blog post. |
| Critic | Evaluates content quality and provides feedback. |
| Editor | Refines and polishes the generated blog content. |
| LinkedIn Agent | Converts the content into a professional LinkedIn post. |
| Twitter/X Agent | Generates a structured Twitter/X thread. |

The critic evaluates the generated content against a configurable quality threshold. If the score is below the threshold, the content is sent back to the writer for revision, subject to the maximum revision limit.

## Technology Stack

- **Python** – Core application development.
- **LangGraph** – Multi-agent workflow orchestration.
- **LangChain** – LLM integration and agent components.
- **Groq** – LLM inference.
- **Tavily** – Web research and information retrieval.
- **Streamlit** – Interactive web application.
- **Pydantic** – Configuration management and data validation.

## Project Structure

```text
ai-content-multi-agent/
│
├── streamlit_app.py
│
├── app/
│   ├── config.py       # Application configuration
│   ├── llm.py          # LLM integration and invocation
│   ├── state.py        # Shared graph state
│   ├── prompts.py      # Agent prompts
│   ├── nodes.py        # Agent implementations and routing
│   ├── graph.py        # LangGraph workflow definition
│   └── progress.py     # Execution progress tracking
│
├── requirements.txt
├── .env
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-content-multi-agent.git

cd ai-content-multi-agent
```

### 2. Create a virtual environment

```bash
python -m venv myvenv
```

Activate the environment:

**Linux/macOS:**

```bash
source myvenv/bin/activate
```

**Windows:**

```bash
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root project directory.

Add the following configuration:

```env
# API Credentials
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key

# Agent Models
RESEARCHER_MODEL=openai/gpt-oss-20b
PLANNER_MODEL=openai/gpt-oss-20b
WRITER_MODEL=openai/gpt-oss-120b
CRITIC_MODEL=openai/gpt-oss-120b
EDITOR_MODEL=openai/gpt-oss-20b
SOCIAL_MODEL=openai/gpt-oss-20b

# Generation Settings
TEMPERATURE=0.4

# Quality Control
MAX_REVISIONS=3
QUALITY_THRESHOLD=8.0
```

Replace the API key placeholders with your own credentials.

**Security:** Never commit your `.env` file or publish API keys on GitHub. Add `.env` and `myvenv/` to `.gitignore`.

### 5. Run the application

```bash
streamlit run streamlit_app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## Configuration Overview

The application uses environment variables to configure the individual agents and content generation workflow.

| Variable | Description |
|---|---|
| `RESEARCHER_MODEL` | Model used by the researcher agent. |
| `PLANNER_MODEL` | Model used by the planner agent. |
| `WRITER_MODEL` | Model used by the writer agent. |
| `CRITIC_MODEL` | Model used by the critic agent. |
| `EDITOR_MODEL` | Model used by the editor agent. |
| `SOCIAL_MODEL` | Model used for social media content generation. |
| `TEMPERATURE` | Controls LLM generation variability. |
| `MAX_REVISIONS` | Maximum number of writer revision attempts. |
| `QUALITY_THRESHOLD` | Minimum quality score required to pass the critic evaluation. |

## Generated Outputs

For each input topic, the application generates:

1. **Blog Post:** Detailed, structured, long-form content.
2. **LinkedIn Post:** Professional content optimized for LinkedIn.
3. **Twitter/X Thread:** A sequence of concise, connected posts.

Generated results are maintained within the browser session. Users can download the generated content using the available download options.

## Author

**Anjna Pal**

Email: [apaul80011@gmail.com](mailto:apaul80011@gmail.com)