from __future__ import annotations
import logging
import re
from typing import List
from openai import AsyncAzureOpenAI
from core.exceptions import QueryGenerationError
from core.interfaces import AIServiceProtocol
from core.models import QueryRequest, SchemaCatalog, SchemaColumn
from utils.config import AppConfig

logger = logging.getLogger(__name__)

class AIService(AIServiceProtocol):
    """Handles all interactions with the Azure OpenAI service."""

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        if not config.azure_api_key:
            raise QueryGenerationError("Azure OpenAI API key must be provided.")

        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
        )

    async def generate_sql_query(
        self,
        request: QueryRequest,
        schema: SchemaCatalog,
    ) -> str:
        prompt = self._build_sql_generation_prompt(request, schema)

        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_query,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Presto SQL generator. "
                            "Use ONLY the provided schema. "
                            "If a field has a formula, use the formula instead of the raw column. "
                            "Return only valid SQL, no explanations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens_sql,
                temperature=self.config.temperature_sql,
            )
        except Exception as exc:
            logger.exception("Azure OpenAI SQL generation failed: %s", exc)
            raise QueryGenerationError(str(exc)) from exc

        sql_query = response.choices[0].message.content.strip()
        return self._clean_sql_query(sql_query)

    async def answer_general_question(self, request: QueryRequest) -> str:
        prompt = (
            "You are an AI assistant for a video analytics platform. "
            "Provide concise, helpful responses about analytics concepts, metrics, or how to work with the system.\n\n"
            f"User question: {request.user_query}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_response,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly AI assistant for a video analytics system. "
                            "Be concise, correct, and avoid hallucinating capabilities the system does not have."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens_response,
                temperature=self.config.temperature_response,
            )
        except Exception as exc:
            logger.exception("Azure OpenAI general response failed: %s", exc)
            raise QueryGenerationError(str(exc)) from exc

        return response.choices[0].message.content.strip()

    def _build_sql_generation_prompt(self, request: QueryRequest, schema: SchemaCatalog) -> str:
        serialized_schema = self._render_schema(schema)
        guidance = [
            "- Use ONLY fields provided in the schema below.",
            "- Match questions to fields using the friendly field name or description.",
            "- If a field has a formula, use the formula instead of the physical column.",
            "- Translate user-facing labels to codes when the description indicates mappings (e.g., platform 4 = IOS).",
            "- Return ONLY a valid Presto SQL query. No comments, no explanations.",
        ]

        if request.previous_error:
            guidance.append(f"- Previous execution error to correct: {request.previous_error}")

        ids_only_hint = ""
        if re.search(r"\b(ids?|identifier|k only|return k)\b", request.user_query, re.IGNORECASE):
            ids_only_hint = (
                "- The user wants only identifier columns. Return only the necessary identifier columns."
            )

        prompt_sections = [
            "Schema definition:",
            serialized_schema,
            "",
            "User question:",
            request.user_query,
            "",
            "Guidance:",
            *guidance,
        ]

        if ids_only_hint:
            prompt_sections.append(ids_only_hint)

        return "\n".join(prompt_sections)

    def _render_schema(self, schema: SchemaCatalog) -> str:
        lines: List[str] = []
        for table_name, table in schema.tables.items():
            lines.append(f"Table: {table_name}")
            base_columns = []
            metrics = []

            for column in table.columns:
                entry = self._format_column(column)
                if column.is_calculated:
                    metrics.append(entry)
                else:
                    base_columns.append(entry)

            if base_columns:
                lines.append("  Base Columns:")
                lines.extend(f"    - {column}" for column in base_columns)
            if metrics:
                lines.append("  Calculated Metrics:")
                lines.extend(f"    - {metric}" for metric in metrics)
            lines.append("")

        return "\n".join(lines).strip()

    def _format_column(self, column: SchemaColumn) -> str:
        description_part = f" - {column.description}" if column.description else ""
        if column.is_calculated and column.formula:
            return f"{column.label} := {column.formula}{description_part}"
        return f"{column.label} -> {column.name} ({column.type}){description_part}"

    def _clean_sql_query(self, sql_query: str) -> str:
        sql_query = re.sub(r"```(sql)?", "", sql_query)
        sql_query = sql_query.strip()
        if sql_query.endswith(";"):
            sql_query = sql_query[:-1]
        return sql_query
