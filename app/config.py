from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# Define all application settings in one place.
class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # ---------------- API keys ----------------
    tavily_api_key: str = ""
    groq_api_key: str = ""

    # ---------------- Models ----------------
    groq_model: str = "openai/gpt-oss-20b"

    researcher_model: str = "openai/gpt-oss-20b"
    planner_model: str = "openai/gpt-oss-20b"
    writer_model: str = "openai/gpt-oss-120b"
    critic_model: str = "openai/gpt-oss-120b"
    editor_model: str = "openai/gpt-oss-20b"
    social_model: str = "openai/gpt-oss-20b"

    # ---------------- Pipeline behaviour ----------------
    temperature: float = 0.4
    max_revisions: int = 3
    quality_threshold: float = 8.0

    # ---------------- Reliability ----------------
    # Plain-text calls: tried in order when the chosen model keeps failing.
    fallback_models: list[str] = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

    # Structured-output calls: only models that support Groq's json_schema mode.
    structured_models: list[str] = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    # gpt-oss models spend output tokens on hidden reasoning. "low" keeps that small.
    # Set to "" to disable.
    reasoning_effort: str = "low"

    request_timeout: float = 90.0
    max_attempts: int = 6          # attempts per model before moving to the next one
    max_rate_wait: float = 75.0    # longer waits (e.g. daily quota) switch model instead

    # Tokens-per-minute limits of your Groq plan (check console.groq.com -> Limits).
    tpm_safety: float = 0.85       # use at most 85% of the limit
    default_tpm_limit: int = 8000
    tpm_limits: dict[str, int] = {
        "openai/gpt-oss-20b": 8000,
        "openai/gpt-oss-120b": 8000,
        "llama-3.3-70b-versatile": 12000,
        "llama-3.1-8b-instant": 6000,
    }

    # ---------------- Output token budgets (per node) ----------------
    # These include the model's hidden reasoning tokens, so keep them generous.
    max_tokens_default: int = 2000
    max_tokens_queries: int = 1000
    max_tokens_research: int = 2200
    max_tokens_planner: int = 2200
    max_tokens_writer: int = 4000
    max_tokens_critic: int = 2200
    max_tokens_editor: int = 4000
    max_tokens_linkedin: int = 1200
    max_tokens_twitter: int = 2000


# Cache the settings object so repeated calls reuse the same configuration.
@lru_cache
def get_settings() -> Settings:
    return Settings()