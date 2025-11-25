from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.models import QueryRequest
from core.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class InteractiveSession:
    """Handles the CLI-based interactive session."""

    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
        self.session_active = True

    async def start(self) -> None:
        self._print_welcome_message()

        while self.session_active:
            user_input = await self._get_user_input()
            if not user_input:
                continue

            if self._should_exit(user_input):
                print("\nGoodbye!")
                self.session_active = False
                break

            request = QueryRequest(user_query=user_input)

            try:
                response = await self.query_processor.process(request)
                print(f"\nAssistant: {response}" if response else "\nAssistant: (no response)")
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                self.session_active = False
                break
            except EOFError:
                print("\nGoodbye!")
                self.session_active = False
                break
            except Exception as exc:
                logger.error("Unexpected error during session: %s", exc, exc_info=True)
                print(f"\nError: {exc}")
                if not await self._prompt_continue():
                    self.session_active = False

    @staticmethod
    def _print_welcome_message() -> None:
        print("\n" + "=" * 60)
        print("AI Query System")
        print("=" * 60)
        print("Ask questions in natural language.")
        print("Type 'quit' to exit.")
        print("=" * 60)

    async def _get_user_input(self) -> str:
        loop = asyncio.get_running_loop()
        try:
            return (await loop.run_in_executor(None, input, "\nYour question: ")).strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        except Exception as exc:
            logger.error("Input error: %s", exc, exc_info=True)
            return ""

    @staticmethod
    def _should_exit(user_input: str) -> bool:
        return user_input.lower() in {"quit", "exit", "q", "bye"}

    async def _prompt_continue(self) -> bool:
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(None, input, "\nContinue session? (y/n): ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    def stop(self) -> None:
        self.session_active = False


class SessionManager:
    """Ensures only one interactive session runs at a time."""

    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
        self.current_session: Optional[InteractiveSession] = None

    async def start_new_session(self) -> None:
        if self.current_session and self.current_session.session_active:
            print("A session is already active. Stopping it...")
            self.stop_current_session()

        self.current_session = InteractiveSession(self.query_processor)
        try:
            await self.current_session.start()
        finally:
            self.current_session = None

    def stop_current_session(self) -> None:
        if self.current_session:
            self.current_session.stop()
            print("Session stopped.")
            self.current_session = None

    def is_session_active(self) -> bool:
        return bool(self.current_session and self.current_session.session_active)
