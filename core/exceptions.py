class AIQueryGeneratorError(Exception):
    """Base exception for the AI Query Generator domain."""


class ConfigurationError(AIQueryGeneratorError):
    """Raised when configuration or environment validation fails."""


class SchemaLoadError(AIQueryGeneratorError):
    """Raised when schema metadata cannot be loaded or parsed."""


class QueryGenerationError(AIQueryGeneratorError):
    """Raised when the AI service fails to produce a valid query."""


class ClassificationError(AIQueryGeneratorError):
    """Raised when the user intent cannot be classified."""


class DatabaseConnectionError(AIQueryGeneratorError):
    """Raised when a connection to the database cannot be established."""


class QueryExecutionError(AIQueryGeneratorError):
    """Raised when executing a generated query fails."""


class CacheError(AIQueryGeneratorError):
    """Raised when caching unexpectedly fails."""
