"""A bounded client for local Ollama inference; no paid service fallback."""

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from src.config import ROOT
from src.interface import Extraction, LABELS, extraction_prompt, review_extraction


class LLMError(RuntimeError):
    """Safe user-facing message without raw server responses."""


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    estimate_usd: StrictInt
    summary: str = Field(min_length=20, max_length=1200)
    limitation: str = Field(min_length=20, max_length=600)


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    enabled: bool = True
    timeout: float = 180

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
            enabled=os.getenv("AMES_LLM_ENABLED", "true").lower() == "true",
            timeout=float(os.getenv("AMES_LLM_TIMEOUT_SECONDS", "180")),
        )


class LocalLLM:
    def __init__(self, settings: Settings | None = None, transport=None):
        self.settings = settings or Settings.from_env()
        parsed = urlparse(self.settings.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "host.docker.internal", "ollama"}:
            raise LLMError("Use a local Ollama address. Remote endpoints are not enabled in this app.")
        if ":cloud" in self.settings.model:
            raise LLMError("Choose a downloaded local model; cloud models are disabled.")
        self.transport = transport
        self.last_usage = {}

    def _json(self, system: str, user: str, schema: type[BaseModel], max_tokens: int):
        if not self.settings.enabled:
            raise LLMError("Language features are switched off. You can still use the manual form.")
        payload = {
            "model": self.settings.model, "stream": False, "think": False, "keep_alive": "5m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "seed": 42, "presence_penalty": 0, "repeat_penalty": 1,
                        "num_ctx": 8192, "num_predict": max_tokens},
        }
        try:
            with httpx.Client(transport=self.transport, timeout=self.settings.timeout, trust_env=False) as client:
                response = client.post(self.settings.base_url + "/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
            if not body.get("done") or body.get("done_reason") == "length":
                raise LLMError("The local model did not finish its response. Try a shorter description.")
            self.last_usage = {k: body.get(k) for k in ("model", "prompt_eval_count", "eval_count", "total_duration")}
            return schema.model_validate_json(body["message"]["content"])
        except httpx.ConnectError as exc:
            raise LLMError("Ollama is not running. Start the local server, then try again.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The local model timed out. Wait for it to finish loading, then try again.") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise LLMError("The selected local model is not installed. Download it with Ollama first.") from exc
            raise LLMError("The local model could not process this request. Please try again.") from exc
        except (httpx.HTTPError, ValueError, ValidationError, KeyError, TypeError) as exc:
            raise LLMError("The language response could not be validated. Please rephrase or use the form.") from exc

    def parse(self, message: str, current: dict, categories: dict):
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            raise LLMError("Enter a home description between 1 and 4,000 characters.")
        content = json.dumps({"previously_confirmed_or_extracted": current, "latest_message": message})
        extraction = self._json(extraction_prompt(categories), content, Extraction, 2200)
        review = review_extraction(extraction, message, current, categories)
        if review.intent == "estimate" and review.missing:
            # One bounded second pass helps the small local model attend to missed
            # details. It can fill only missing fields, and repeats all validation.
            targets = review.missing
            repair_prompt = extraction_prompt(categories) + (
                " This is a focused second pass. Extract ONLY these fields if explicitly stated: "
                + json.dumps(targets) + ". Leave all other fields out. Copy evidence verbatim, "
                "including the number and unit for measurements. Do not insert extra words into "
                "quotes. If a requested field is absent or ambiguous, omit it and ask for it. "
                "Never change a stated number to make it valid."
            )
            first_usage = self.last_usage
            repaired = self._json(repair_prompt, json.dumps({"latest_message": message}), Extraction, 1800)
            if any(item.name not in targets for item in repaired.updates):
                return review
            revised = review_extraction(repaired, message, review.features, categories)
            revised.issues.extend(issue for issue in review.issues if not any(
                key in issue or LABELS.get(key, key) in issue
                for key in targets if key in revised.features
            ))
            self.last_usage = {"passes": [first_usage, self.last_usage]}
            return revised
        return review

    def explain(self, price: float, metadata: dict) -> Explanation:
        rounded = round(price)
        system = (
            "Write a short, plain-English explanation of the supplied trained housing model result. "
            "Return JSON matching the supplied schema. Echo estimate_usd EXACTLY. Include the "
            "formatted dollar estimate in summary. Explain that this is a model estimate of a "
            "historical sale price, not an actual sale or present appraisal. In limitation mention "
            "Ames, 2006–2010 and that individual errors may exceed the average test error. "
            "Do not invent feature contributions, confidence intervals, market trends or advice. "
            "Only discuss the supplied facts. Do not treat MAE or R² as a confidence percentage."
        )
        facts = {"estimate_usd": rounded, "display_estimate": f"${rounded:,}",
                 "location": "Ames, Iowa", "sale_years": "2006–2010", "model": "Random Forest",
                 "test_mae_usd": round(metadata["test_metrics"]["mae"]), "test_rows": metadata["test_rows"]}
        result = self._json(system, json.dumps(facts), Explanation, 650)
        if result.estimate_usd != rounded or f"${rounded:,}" not in result.summary:
            raise LLMError("The language explanation misstated the estimate, so it was withheld.")
        return result
