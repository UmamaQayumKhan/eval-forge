"""Dataset loading, slicing and integrity checks.

The integrity checks exist because a quietly broken golden set produces
confident, meaningless numbers. Duplicate inputs inflate a score by weighting
the same question twice; an item with no expectation at all is scored as a free
pass; a set with no negative cases cannot measure over-confidence.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Item:
    id: str
    input: str
    expected: Any = None
    # Per-item expectations, consumed by the matching scorer.
    contains_any: list[str] = field(default_factory=list)
    contains_all: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    rubric: str = ""
    reference: str = ""
    # Tri-state on purpose. True = must refuse, False = must NOT refuse,
    # None = do not assert either way.
    #
    # A two-state bool conflated "refused to answer" with "declined to act".
    # A reply like "I'm not able to process refunds myself, I'll pass this to a
    # colleague" correctly declines to act while still answering, and with a
    # default of False the refusal scorer marked it an over-refusal. Items that
    # genuinely care now say so explicitly.
    should_refuse: bool | None = None
    required_keys: list[str] = field(default_factory=list)
    n_sources: int = 0
    numeric: float | None = None
    # Metadata
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    # Optional human label, used to validate an LLM judge with Cohen's kappa.
    human_label: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat view handed to scorers."""
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "contains_any": self.contains_any,
            "contains_all": self.contains_all,
            "must_not_contain": self.must_not_contain,
            "rubric": self.rubric,
            "reference": self.reference,
            "should_refuse": self.should_refuse,
            "required_keys": self.required_keys,
            "n_sources": self.n_sources,
            "numeric": self.numeric,
            "category": self.category,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "human_label": self.human_label,
            **self.extra,
        }

    @property
    def has_expectation(self) -> bool:
        return bool(
            self.expected
            or self.contains_any
            or self.contains_all
            or self.must_not_contain
            or self.rubric
            or self.required_keys
            or self.should_refuse
            or self.numeric is not None
        )


KNOWN_FIELDS = set(Item.__dataclass_fields__) - {"extra"}


@dataclass
class Dataset:
    name: str
    items: list[Item] = field(default_factory=list)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    @property
    def fingerprint(self) -> str:
        """Hash of the inputs and expectations.

        Stored in every baseline. A regression gate that compares against a
        baseline taken on a *different* dataset is worse than no gate, and
        without a fingerprint that mistake is invisible.
        """
        hasher = hashlib.blake2b(digest_size=16)
        for item in sorted(self.items, key=lambda i: i.id):
            hasher.update(item.id.encode())
            hasher.update(item.input.encode())
            hasher.update(json.dumps(item.as_dict(), sort_keys=True, default=str).encode())
        return hasher.hexdigest()

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return dict(sorted(counts.items()))

    def filter(self, *, category: str | None = None, tag: str | None = None) -> "Dataset":
        items = self.items
        if category:
            items = [i for i in items if i.category == category]
        if tag:
            items = [i for i in items if tag in i.tags]
        return Dataset(name=f"{self.name}[{category or tag}]", items=items, path=self.path)

    def sample(self, n: int, seed: int = 20260101) -> "Dataset":
        """Deterministic subsample, stratified by category so a quick run still
        covers every category instead of 30 billing questions."""
        if n >= len(self.items):
            return self
        rng = random.Random(seed)
        by_cat: dict[str, list[Item]] = {}
        for item in self.items:
            by_cat.setdefault(item.category, []).append(item)

        picked: list[Item] = []
        cats = sorted(by_cat)
        per_cat = max(1, n // max(1, len(cats)))
        for cat in cats:
            pool = sorted(by_cat[cat], key=lambda i: i.id)
            rng.shuffle(pool)
            picked.extend(pool[:per_cat])

        remaining = [i for i in self.items if i not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: max(0, n - len(picked))])
        return Dataset(name=f"{self.name}[sample={n}]", items=picked[:n], path=self.path)

    def validate(self) -> list[str]:
        """Integrity problems, as a list of human-readable strings."""
        problems: list[str] = []

        if not self.items:
            return ["dataset is empty"]

        ids = [i.id for i in self.items]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            problems.append(f"duplicate item ids: {dupes[:5]}")

        seen: dict[str, str] = {}
        for item in self.items:
            key = " ".join(item.input.lower().split())
            if key in seen:
                problems.append(
                    f"duplicate input: {item.id!r} repeats {seen[key]!r} "
                    "(double-weights the same question)"
                )
            else:
                seen[key] = item.id

        no_expectation = [i.id for i in self.items if not i.has_expectation]
        if no_expectation:
            problems.append(
                f"items with no expectation (scored as free passes): {no_expectation[:5]}"
            )

        blank = [i.id for i in self.items if not i.input.strip()]
        if blank:
            problems.append(f"items with blank input: {blank[:5]}")

        negatives = [i for i in self.items if i.should_refuse]
        if not negatives:
            problems.append(
                "no negative cases: a system that answers everything will score "
                "perfectly and be unsafe. Add items with should_refuse=true."
            )
        elif len(negatives) / len(self.items) < 0.05:
            problems.append(
                f"only {len(negatives)}/{len(self.items)} negative cases; "
                "refusal accuracy will be estimated from very little data"
            )

        if len(self.items) < 30:
            problems.append(
                f"only {len(self.items)} items; confidence intervals will be very "
                "wide (see `evalforge power`)"
            )

        return problems


def load_jsonl(path: str | Path, name: str | None = None) -> Dataset:
    p = Path(path)
    items: list[Item] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name}:{lineno}: invalid JSON — {exc.msg}") from exc

        known = {k: v for k, v in raw.items() if k in KNOWN_FIELDS}
        extra = {k: v for k, v in raw.items() if k not in KNOWN_FIELDS}
        known.setdefault("id", f"item-{lineno:04d}")
        if "input" not in known:
            raise ValueError(f"{p.name}:{lineno}: item is missing required field 'input'")
        items.append(Item(**known, extra=extra))

    return Dataset(name=name or p.stem, items=items, path=p)


def load_many(paths: list[str | Path]) -> Dataset:
    """Merge several files, namespacing ids by filename so two files cannot
    collide and silently drop items in the duplicate check."""
    out: list[Item] = []
    names: list[str] = []
    for path in paths:
        ds = load_jsonl(path)
        names.append(ds.name)
        for item in ds.items:
            item.id = f"{ds.name}/{item.id}"
            out.append(item)
    return Dataset(name="+".join(names), items=out)
