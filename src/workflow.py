"""Review state shared by the app and its real-model workflow evaluation.

The editable draft and the last prediction are deliberately separate snapshots.
Sources describe how a value arrived, not whether it is factually correct.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from src.interface import LABELS, Review, validate_features
from src.serving import predict


@dataclass
class HomeWorkflow:
    features: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)
    field_issues: dict = field(default_factory=dict)
    general_issues: list = field(default_factory=list)
    intent: str = "estimate"
    question: str = ""
    error: str = ""
    result: dict | None = None
    revision: int = 0

    def invalidate(self):
        self.revision += 1

    @property
    def stale(self):
        return self.result is not None and self.result["revision"] != self.revision

    def apply(self, review: Review):
        """Apply only this turn's attempted fields; keep unrelated manual corrections."""
        self.invalidate()
        self.intent, self.question, self.error = review.intent, review.question, ""
        self.changes = {}
        if review.intent == "out_of_scope":
            # Never show an earlier Ames result as an answer to a different city.
            self.result = None
            return
        for key in review.touched:
            if key not in LABELS:
                continue
            before, after = self.features.get(key), review.features.get(key)
            if after is None:
                self.features.pop(key, None)
                self.sources.pop(key, None)
            else:
                self.features[key] = after
                self.sources[key] = {"source": "From your description", **review.evidence.get(key, {})}
            if before != after:
                self.changes[key] = {"before": before, "after": after}
            self.field_issues.pop(key, None)
        self.field_issues.update({key: deepcopy(items) for key, items in review.field_issues.items() if key in LABELS})
        mapped = [issue for key, items in review.field_issues.items() if key in LABELS for issue in items]
        self.general_issues = [issue for issue in review.issues if issue not in mapped]

    def edit(self, key, value):
        before = self.features.get(key)
        if before == value and key not in self.field_issues:
            return
        self.invalidate()
        if value is None:
            self.features.pop(key, None)
            self.sources.pop(key, None)
        else:
            self.features[key] = value
            self.sources[key] = {"source": "Edited by you"}
        original = self.changes.get(key, {}).get("before", before)
        if original == value:
            self.changes.pop(key, None)
        else:
            self.changes[key] = {"before": original, "after": value}
        self.field_issues.pop(key, None)

    def fail_parsing(self, message):
        self.invalidate()
        # A failed request cannot undo a previous out-of-scope decision.
        if self.intent != "out_of_scope":
            self.intent = "clarify"
        self.error = message

    def use_manual_form(self):
        """An explicit user action resolves a general language question, not bad fields."""
        if self.intent == "out_of_scope":
            return
        self.intent, self.question, self.error = "estimate", "", ""
        self.general_issues = []
        self.invalidate()

    def review(self, categories):
        review = validate_features(self.features, categories)
        for key, items in self.field_issues.items():
            review.field_issues.setdefault(key, []).extend(items)
            review.issues.extend(items)
        review.issues.extend(self.general_issues)
        review.intent, review.question = self.intent, self.question
        return review

    def estimate(self, model, metadata, defaults, *, confirmed, language_enabled):
        review = self.review(metadata["categories"])
        if not confirmed or not review.ready or self.error:
            raise ValueError("Review and confirm the required details and resolve the flagged inputs first.")
        self.result = {
            "id": uuid4().hex, "revision": self.revision,
            "price": predict(model, metadata, review.features),
            "features": deepcopy(review.features), "sources": deepcopy(self.sources),
            "defaults": {key: defaults[key] for key in review.defaulted},
            "explanation": None, "error": None,
            "explanation_status": "pending" if language_enabled else "manual",
        }
        self.changes = {}
        return self.result

    def save_explanation(self, result_id, *, explanation=None, error=None):
        # A response arriving after an edit/reset must never attach to a new estimate.
        if not self.result or self.result["id"] != result_id or self.stale:
            return False
        self.result.update(explanation=explanation, error=error,
                           explanation_status="failed" if error else "complete")
        return True
