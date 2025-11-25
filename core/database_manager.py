from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import prestodb
from prestodb import exceptions as presto_exceptions

from core.exceptions import (
    DatabaseConnectionError,
    QueryExecutionError,
    SchemaLoadError,
)
from core.interfaces import DatabaseExecutorProtocol
from core.models import QueryResult, SchemaCatalog, SchemaColumn, SchemaTable
from utils.config import AppConfig

logger = logging.getLogger(__name__)


class DatabaseManager(DatabaseExecutorProtocol):
    """Manages Presto connectivity, schema loading, and query history."""

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        self.config = config
        self._connection = None
        self._cursor = None
        self._lock = threading.RLock()
        self._schema_loaded = asyncio.Event()
        self._schema_load_lock = asyncio.Lock()
        self._loaded_schema: Optional[SchemaCatalog] = None

    async def execute(
        self,
        sql: str,
        *,
        user_question: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        query_id = (metadata or {}).get("query_id") or uuid4()

        if not self.config.execute_queries:
            logger.info("Execution disabled via configuration; returning SQL preview only.")
            return QueryResult(query_id=query_id, sql=sql, rows=[], columns=[], row_count=0, cached=False)

        return await asyncio.to_thread(
            self._execute_sync,
            sql,
            user_question,
            metadata or {},
        )

    def _execute_sync(
        self,
        sql: str,
        user_question: Optional[str],
        metadata: Dict[str, Any],
    ) -> QueryResult:
        query_id: UUID = metadata.get("query_id") or uuid4()
        start_ts = datetime.utcnow()

        try:
            self._ensure_connection()
            assert self._cursor is not None
            self._cursor.execute(sql)

            if self._cursor.description is None:
                return QueryResult(
                    query_id=query_id,
                    sql=sql,
                    rows=[],
                    columns=[],
                    row_count=0,
                    execution_ms=_duration_ms(start_ts),
                )

            rows = self._cursor.fetchall()
            columns = [desc[0] for desc in self._cursor.description]
            result_rows = [dict(zip(columns, row)) for row in rows]

            return QueryResult(
                query_id=query_id,
                sql=sql,
                rows=result_rows,
                columns=columns,
                row_count=len(result_rows),
                execution_ms=_duration_ms(start_ts),
            )
        except presto_exceptions.PrestoUserError as exc:
            logger.exception("Presto user error: %s", exc)
            raise QueryExecutionError(str(exc)) from exc
        except presto_exceptions.PrestoQueryError as exc:
            logger.exception("Presto query error: %s", exc)
            raise QueryExecutionError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Query execution failed: %s", exc)
            raise QueryExecutionError(str(exc)) from exc

    def _ensure_connection(self) -> None:
        with self._lock:
            if self._connection and self._cursor:
                return
            try:
                self._connection = prestodb.dbapi.connect(
                    host=self.config.presto_host,
                    port=self.config.presto_port,
                    user=self.config.presto_user,
                    catalog=self.config.presto_catalog,
                    schema=self.config.presto_schema,
                    http_scheme="http",
                )
                self._cursor = self._connection.cursor()
                logger.info("Connected to Presto at %s:%s", self.config.presto_host, self.config.presto_port)
            except Exception as exc:
                raise DatabaseConnectionError(str(exc)) from exc

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            try:
                if self._cursor:
                    self._cursor.close()
                if self._connection:
                    self._connection.close()
            except Exception as exc:
                logger.warning("Error closing Presto connection: %s", exc)
            finally:
                self._cursor = None
                self._connection = None

    async def get_schema(self) -> SchemaCatalog:
        async with self._schema_load_lock:
            if self._loaded_schema is not None:
                return self._loaded_schema

            schema = await asyncio.to_thread(self._load_schema_from_disk, self.config.schema_json_path)
            self._loaded_schema = schema
            self._schema_loaded.set()
            return schema

    async def wait_for_schema(self) -> SchemaCatalog:
        await self._schema_loaded.wait()
        if self._loaded_schema is None:
            raise SchemaLoadError("Schema not loaded after waiting.")
        return self._loaded_schema

    def _load_schema_from_disk(self, path: Path) -> SchemaCatalog:
        if not path.exists():
            raise SchemaLoadError(f"Schema file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            raise SchemaLoadError(f"Failed to parse schema file: {exc}") from exc

        if not isinstance(data, dict):
            raise SchemaLoadError(
                "Schema JSON must be a dictionary keyed by table name with a 'columns' list."
            )
        return self._build_catalog_from_dict(data)

    def _build_catalog_from_dict(self, data: Dict[str, Any]) -> SchemaCatalog:
        catalog = SchemaCatalog()
        for table_name, table_data in data.items():
            columns: List[SchemaColumn] = []
            for column_data in table_data.get("columns", []):
                if not isinstance(column_data, dict):
                    continue
                column = SchemaColumn(
                    name=column_data.get("name", "unknown"),
                    type=column_data.get("type", "unknown"),
                    field_name=column_data.get("field_name"),
                    description=column_data.get("description"),
                    formula=column_data.get("formula"),
                )
                columns.append(column)
            catalog.add_table(SchemaTable(name=table_name, columns=columns))
        return catalog


def _duration_ms(start_time: datetime) -> int:
    delta = datetime.utcnow() - start_time
    return int(delta.total_seconds() * 1000)
