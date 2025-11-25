from __future__ import annotations
import asyncio
import logging
import typer
from core.container import ApplicationContainer
from core.interactive_session import SessionManager
from core.models import QueryRequest
from core.exceptions import ConfigurationError
from utils.config import AppConfig, load_config

app = typer.Typer(add_completion=False, help="AI-powered query generation CLI.")


def configure_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.log_file),
        ],
    )


@app.command()
def interactive() -> None:
    """Start an interactive CLI session."""

    async def _run() -> None:
        config = load_config()
        configure_logging(config)
        container = ApplicationContainer(config)
        async with container.lifespan() as processor:
            session_manager = SessionManager(processor)
            await session_manager.start_new_session()

    try:
        asyncio.run(_run())
    except ConfigurationError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")


@app.command()
def query(text: str = typer.Argument(..., help="Natural language question to convert to SQL.")) -> None:
    """Run a single query end-to-end."""

    async def _run() -> str:
        config = load_config()
        configure_logging(config)
        container = ApplicationContainer(config)
        async with container.lifespan() as processor:
            request = QueryRequest(user_query=text)
            return await processor.process(request)

    try:
        result = asyncio.run(_run())
        typer.echo(result)
    except ConfigurationError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")


@app.command()
def health() -> None:
    """Perform a basic health check of external dependencies."""

    async def _run() -> str:
        config = load_config()
        configure_logging(config)
        container = ApplicationContainer(config)
        try:
            schema = await container.database.get_schema()
            status = "ok" if schema.tables else "warning: schema loaded but empty"
        finally:
            await container.shutdown()
        return status

    try:
        status = asyncio.run(_run())
        typer.echo(f"Health check: {status}")
    except ConfigurationError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Health check failed: {exc}")
        raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """Entry point for the CLI."""


if __name__ == "__main__":
    app()
