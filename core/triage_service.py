from __future__ import annotations
import logging
from openai import AsyncAzureOpenAI
from core.exceptions import ClassificationError
from core.interfaces import TriageServiceProtocol
from core.models import QueryRequest
from utils.config import AppConfig

logger = logging.getLogger(__name__)

GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "yo",
    "hola",
    "namaste",
}

class TriageService(TriageServiceProtocol):
    """Classifies user queries and routes them appropriately."""

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
        )

    async def classify_query(self, request: QueryRequest) -> str:
        lower_query = request.user_query.lower()
        if any(keyword in lower_query for keyword in GREETING_KEYWORDS):
            return "GREETING"

        prompt = (
            "Classify the user query into one of four categories:\n"
            "1. DATA_QUESTION: Questions that require querying data, metrics, analytics, statistics.\n"
            "2. GENERAL_QUESTION: Questions about how the system works, definitions, explanations, or high-level concepts.\n"
            "3. GREETING: Salutations such as hi, hello, good morning.\n"
            "4. OUT_OF_SCOPE: Anything unrelated to the analytics system.\n\n"
            f"User query: {request.user_query}\n\n"
            "Respond with exactly one label: DATA_QUESTION, GENERAL_QUESTION, GREETING, or OUT_OF_SCOPE."
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_classification,
                messages=[
                    {
                        "role": "system",
                        "content": "You classify user queries. Reply with exactly one label.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.0,
            )
        except Exception as exc:
            logger.exception("Classification request failed: %s", exc)
            raise ClassificationError(str(exc)) from exc

        label = (response.choices[0].message.content or "").strip().upper()
        if label not in {"DATA_QUESTION", "GENERAL_QUESTION", "GREETING", "OUT_OF_SCOPE"}:
            logger.warning("Unexpected classification '%s'; defaulting to DATA_QUESTION", label)
            label = "DATA_QUESTION"

        return label
