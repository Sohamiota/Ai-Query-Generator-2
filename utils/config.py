from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, FieldValidationInfo, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigurationError


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables or .env files."""

    app_env: str = Field(default="dev", env="APP_ENV")

    presto_host: str = Field(..., env="PRESTO_HOST")
    presto_port: int = Field(default=8090, env="PRESTO_PORT")
    presto_user: str = Field(default="hadoop", env="PRESTO_USER")
    presto_catalog: str = Field(default="hive", env="PRESTO_CATALOG")
    presto_schema: str = Field(default="test", env="PRESTO_SCHEMA")

    azure_api_key: str = Field(..., env="AZURE_API_KEY")
    azure_endpoint: str = Field(..., env="AZURE_ENDPOINT")
    azure_api_version: str = Field(default="2025-01-01-preview", env="AZURE_API_VERSION")

    deployment_query: str = Field(default="gpt-4.1", env="AZURE_DEPLOYMENT_QUERY")
    deployment_classification: str = Field(default="gpt-4.1", env="AZURE_DEPLOYMENT_CLASSIFICATION")
    deployment_response: str = Field(default="gpt-4.1", env="AZURE_DEPLOYMENT_RESPONSE")

    max_tokens_sql: int = Field(default=800, env="MAX_TOKENS_SQL")
    max_tokens_response: int = Field(default=500, env="MAX_TOKENS_RESPONSE")
    temperature_sql: float = Field(default=0.3, env="TEMPERATURE_SQL")
    temperature_response: float = Field(default=0.5, env="TEMPERATURE_RESPONSE")

    max_retries: int = Field(default=2, env="MAX_RETRIES")
    max_results_display: int = Field(default=5, env="MAX_RESULTS_DISPLAY")

    log_file: str = Field(default="ai_rag_presto.log", env="LOG_FILE")
    schema_json_path: Path = Field(default=Path("data/mapping.json"), env="SCHEMA_JSON_PATH")
    query_history_path: Path = Field(default=Path("query_history.json"), env="QUERY_HISTORY_PATH")

    cache_ttl_schema_seconds: int = Field(default=900, env="CACHE_TTL_SCHEMA_SECONDS")
    cache_ttl_classification_seconds: int = Field(default=300, env="CACHE_TTL_CLASSIFICATION_SECONDS")

    execute_queries: bool = Field(default=True, env="EXECUTE_QUERIES")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    @field_validator("azure_api_key", "azure_endpoint")
    def _must_not_be_empty(cls, value: str, info: FieldValidationInfo) -> str:
        if not value:
            raise ConfigurationError(f"{info.field_name} must be provided.")
        return value

    @field_validator("schema_json_path", "query_history_path", mode="before")
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()

    @field_validator("execute_queries", mode="before")
    def _parse_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @property
    def schema_path_exists(self) -> bool:
        return self.schema_json_path.exists()

    def ensure_valid(self) -> None:
        """Raise ConfigurationError if any critical configuration is missing."""
        missing = []
        if not self.azure_api_key:
            missing.append("AZURE_API_KEY")
        if not self.azure_endpoint:
            missing.append("AZURE_ENDPOINT")
        if not self.presto_host:
            missing.append("PRESTO_HOST")
        if missing:
            raise ConfigurationError(f"Missing required configuration values: {', '.join(missing)}")


@lru_cache(maxsize=1)
def load_config(**overrides: object) -> AppConfig:
    """Load configuration once per process."""
    try:
        config = AppConfig(**overrides)
    except ValidationError as exc:
        missing = [
            ".".join(str(part) for part in error.get("loc", []))
            for error in exc.errors()
            if error.get("type") == "missing"
        ]
        detail = f"Missing required configuration values: {', '.join(missing)}" if missing else str(exc)
        raise ConfigurationError(
            f"{detail}. Provide them via environment variables or a .env file."
        ) from exc
    config.ensure_valid()
    return config


def reset_config_cache() -> None:
    """Primarily for testing; clears cached configuration."""
    load_config.cache_clear()  # type: ignore[attr-defined]
