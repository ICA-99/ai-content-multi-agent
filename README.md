# Content Pipeline Agent

An AI-powered content generation system that transforms a single topic into three distinct content assets: a **blog post, a LinkedIn post, and a Twitter/X thread**.

The application is built using **LangGraph** for agent orchestration and **Streamlit** for the user interface. It follows a multi-agent workflow in which specialized agents collaborate to research, plan, generate, evaluate, and refine content.

## Live Demo

**Try the application:** [Content Pipeline Agent – Live Demo](https://ai-content-multi-agent-jhkvvkovpmmmqr7v7lpmgh.streamlit.app/)

> **Note:** Content generation may take some time. The application uses multiple LLM-powered agents, and requests are processed through several stages. To manage token usage, reduce the likelihood of API rate-limit errors, and maintain reliable execution, the workflow incorporates pauses between certain LLM calls. Please allow some time for the complete generation process.

## Workflow Architecture

The application uses a LangGraph-based workflow to coordinate multiple specialized agents.

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
  +---- Score acceptable / Maximum rounds reached
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
| Researcher | Gathers relevant information about the given topic. |
| Planner | Develops the content structure and writing strategy. |
| Writer | Generates the initial blog content. |
| Critic | Evaluates content quality and provides feedback. |
| Editor | Refines and polishes the approved content. |
| LinkedIn Agent | Converts the content into a professional LinkedIn post. |
| Twitter/X Agent | Generates a structured Twitter/X thread. |

The critic can send the content back to the writer for revision when the quality score falls below the configured threshold, subject to the maximum revision limit.

## Technology Stack

- **Python** – Core application development.
- **LangGraph** – Multi-agent workflow orchestration.
- **LangChain** – LLM integration and agent components.
- **Groq** – LLM inference.
- **Tavily** – Web research and information retrieval.
- **Streamlit** – Interactive web application.
- **Pydantic** – Data validation and structured outputs.
- **Pytest** – Automated testing.

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
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── .env.example
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
python -m venv .venv
```

Activate it:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using the provided example:

```bash
cp .env.example .env
```

Configure the required API credentials:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The Tavily API key is optional if web research is not required by your configuration.

### 5. Run the application

```bash
streamlit run streamlit_app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## Running Tests

Execute the test suite using:

```bash
pytest -q
```

## Output and Data Handling

The application generates three content assets:

1. **Blog Post** – Structured, informative, and detailed long-form content.
2. **LinkedIn Post** – Professional content formatted for LinkedIn.
3. **Twitter/X Thread** – A sequence of concise, connected posts.

Generated results are maintained within the browser session. Users can download the generated content using the available download options.

## Author

**Anjna Pal**

Email: [apaul80011@gmail.com](mailto:apaul80011@gmail.com)