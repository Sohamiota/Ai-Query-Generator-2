from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from core.ai_service import AIService
from core.database_manager import DatabaseManager
from core.query_processor import QueryProcessor
from core.triage_service import TriageService
from utils.config import AppConfig, load_config


class ApplicationContainer:
    """Lightweight dependency container for assembling application services."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()

        self._database = DatabaseManager(self.config)
        self._ai_service = AIService(self.config)
        self._triage_service = TriageService(self.config)
        self._query_processor = QueryProcessor(
            ai_service=self._ai_service,
            triage_service=self._triage_service,
            database=self._database,
            config=self.config,
        )

    @property
    def query_processor(self) -> QueryProcessor:
        return self._query_processor

    @property
    def database(self) -> DatabaseManager:
        return self._database

    async def shutdown(self) -> None:
        await self._database.close()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[QueryProcessor]:
        try:
            await self._database.get_schema()
        except Exception:
            # Schema loading errors will surface when first requested;
            # we swallow them here to avoid blocking startup.
            pass
        try:
            yield self._query_processor
        finally:
            await self.shutdown()
