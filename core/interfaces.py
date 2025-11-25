from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from core.models import QueryRequest, QueryResult, SchemaCatalog

@runtime_checkable
class CacheProtocol(Protocol):
    """Minimal cache interface used by the services."""

    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


@runtime_checkable
class AIServiceProtocol(Protocol):
    """Service responsible for interacting with the LLM."""

    async def generate_sql_query(
        self,
        request: QueryRequest,
        schema: SchemaCatalog,
    ) -> str:
        ...

    async def answer_general_question(self, request: QueryRequest) -> str:
        ...


@runtime_checkable
class TriageServiceProtocol(Protocol):
    """Classifies user intent before execution."""

    async def classify_query(self, request: QueryRequest) -> str:
        ...


@runtime_checkable
class DatabaseExecutorProtocol(Protocol):
    """Executes SQL against the underlying database."""

    async def execute(
        self,
        sql: str,
        *,
        user_question: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        ...

    async def get_schema(self) -> SchemaCatalog:
        ...


@runtime_checkable
class QueryGeneratorProtocol(Protocol):
    """Generates SQL (or other query languages) from user requests."""

    async def generate(self, request: QueryRequest, schema: SchemaCatalog) -> str:
        ...


@runtime_checkable
class QueryProcessorProtocol(Protocol):
    """High level orchestrator used by the interactive session or API."""

    async def process(self, request: QueryRequest) -> str:
        ...
