from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class SchemaColumn(BaseModel):
    """Represents a column or calculated metric that can appear in the schema."""

    name: str
    type: str = Field(default="text", description="Database type, e.g., varchar, bigint")
    field_name: Optional[str] = Field(
        default=None, description="Friendly name exposed to the user"
    )
    description: Optional[str] = Field(default=None)
    formula: Optional[str] = Field(
        default=None,
        description="Formula to use instead of the raw column, when provided",
    )

    @property
    def label(self) -> str:
        return self.field_name or self.name

    @property
    def is_calculated(self) -> bool:
        return bool(self.formula)


class SchemaTable(BaseModel):
    """Holds schema metadata for a physical table."""

    name: str
    columns: List[SchemaColumn] = Field(default_factory=list)

    @validator("columns", each_item=True)
    def _ensure_schema_column(cls, value: SchemaColumn) -> SchemaColumn:
        if not isinstance(value, SchemaColumn):
            raise TypeError("columns must contain SchemaColumn instances")
        return value

    def get_column(self, name: str) -> Optional[SchemaColumn]:
        lowered = name.lower()
        for col in self.columns:
            if col.name.lower() == lowered or (col.field_name and col.field_name.lower() == lowered):
                return col
        return None


class SchemaCatalog(BaseModel):
    """Top-level schema container that can describe multiple tables."""

    tables: Dict[str, SchemaTable] = Field(default_factory=dict)

    def get_table(self, name: str) -> Optional[SchemaTable]:
        return self.tables.get(name) or self.tables.get(name.lower())

    def add_table(self, table: SchemaTable) -> None:
        self.tables[table.name] = table

    def all_columns(self) -> Iterable[SchemaColumn]:
        for table in self.tables.values():
            yield from table.columns


class QueryRequest(BaseModel):
    """Represents a request to generate or execute a query."""

    id: UUID = Field(default_factory=uuid4)
    user_query: str
    previous_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("user_query")
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_query cannot be empty")
        return stripped


class QueryResult(BaseModel):
    """Structured representation of a query execution result."""

    query_id: UUID
    sql: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    execution_ms: Optional[int] = None
    cached: bool = False

    @validator("row_count", always=True)
    def _default_row_count(cls, value: int, values: Dict[str, Any]) -> int:
        if value:
            return value
        rows = values.get("rows") or []
        return len(rows)


class QueryError(BaseModel):
    """Details about a failed query attempt."""

    query_id: UUID
    sql: Optional[str] = None
    error_type: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryHistoryRecord(BaseModel):
    """Record persisted for query history or auditing."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_id: UUID
    user: str
    sql: str
    success: bool
    row_count: int
    user_question: Optional[str] = None
    schema_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
