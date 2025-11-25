from __future__ import annotations

import logging
from core.exceptions import (
    ClassificationError,
    DatabaseConnectionError,
    QueryExecutionError,
    QueryGenerationError,
    SchemaLoadError,
)
from core.interfaces import (
    AIServiceProtocol,
    DatabaseExecutorProtocol,
    QueryProcessorProtocol,
    TriageServiceProtocol,
)
from core.models import QueryRequest, QueryResult
from utils.config import AppConfig

logger = logging.getLogger(__name__)


class QueryProcessor(QueryProcessorProtocol):
    """High-level orchestrator for the AI Query Generator workflow."""

    def __init__(
        self,
        *,
        ai_service: AIServiceProtocol,
        triage_service: TriageServiceProtocol,
        database: DatabaseExecutorProtocol,
        config: AppConfig,
    ) -> None:
        self.ai_service = ai_service
        self.triage_service = triage_service
        self.database = database
        self.config = config

    async def process(self, request: QueryRequest) -> str:
        logger.info("Processing query %s: %s", request.id, request.user_query)
        try:
            category = await self.triage_service.classify_query(request)
        except ClassificationError as exc:
            logger.exception("Failed to classify query: %s", exc)
            return "I could not determine how to handle that question. Please try rephrasing."

        if category == "OUT_OF_SCOPE":
            return (
                "I'm sorry, but that question is outside my current scope. "
                "I can help with video analytics data questions or information about the system."
            )

        if category in {"GREETING", "GENERAL_QUESTION"}:
            return await self._handle_general_question(request)

        if category == "DATA_QUESTION":
            return await self._handle_data_question(request)

        logger.warning("Received unexpected category '%s'; falling back to data pipeline.", category)
        return await self._handle_data_question(request)

    async def _handle_general_question(self, request: QueryRequest) -> str:
        try:
            return await self.ai_service.answer_general_question(request)
        except QueryGenerationError as exc:
            logger.exception("Failed to generate conversational response: %s", exc)
            return "I ran into an issue while answering that. Please try again shortly."

    async def _handle_data_question(self, request: QueryRequest) -> str:
        try:
            schema = await self.database.get_schema()  # type: ignore[attr-defined]
        except AttributeError:
            raise QueryGenerationError("Database manager does not expose schema retrieval.")  # pragma: no cover
        except SchemaLoadError as exc:
            logger.exception("Failed to load schema: %s", exc)
            return (
                "I couldn't load the schema information required to generate SQL. "
                "Please verify the schema configuration and try again."
            )
        except Exception as exc:
            logger.exception("Unexpected error loading schema: %s", exc)
            return (
                "Something went wrong while loading the schema definition. "
                "Please try again or contact support."
            )

        try:
            sql = await self.ai_service.generate_sql_query(request, schema)
        except QueryGenerationError as exc:
            logger.exception("SQL generation failed: %s", exc)
            return (
                "I wasn't able to generate a valid SQL query for that question. "
                "Please revise the question or provide more detail."
            )

        try:
            result = await self.database.execute(
                sql,
                user_question=request.user_query,
                metadata={"query_id": request.id},
            )
        except QueryExecutionError as exc:
            logger.exception("SQL execution failed: %s", exc)
            return (
                f"Generated SQL Query:\n\n{sql}\n\n"
                "The query failed to execute. Please review the SQL and try again."
            )
        except DatabaseConnectionError as exc:  # type: ignore[name-defined]
            logger.exception("Database connection failed: %s", exc)
            return "Unable to connect to the database right now. Please try again later."
        except Exception as exc:
            logger.exception("Unexpected database error: %s", exc)
            return (
                f"Generated SQL Query:\n\n{sql}\n\n"
                "An unexpected error occurred while executing the query."
            )

        return self._format_success_response(sql, result)

    def _format_success_response(self, sql: str, result: QueryResult) -> str:
        if not result.rows:
            return f"Generated SQL Query:\n\n{sql}\n\nNo results were returned."

        preview_rows = result.rows[: self.config.max_results_display]
        output_lines = [
            "Generated SQL Query:",
            "",
            sql,
            "",
            f"Results ({result.row_count} row{'s' if result.row_count != 1 else ''}):",
            "",
        ]
        for idx, row in enumerate(preview_rows, start=1):
            output_lines.append(f"{idx}. {row}")

        remaining = result.row_count - len(preview_rows)
        if remaining > 0:
            output_lines.append("")
            output_lines.append(f"... and {remaining} more row(s).")

        return "\n".join(output_lines)
