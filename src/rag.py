"""
Retrieval-Augmented recommendation for VibeMatch.

A listener types a free-text request ("something calm for studying"). The pipeline:

  1. GUARDRAILS   validate the request (type, length, content)
  2. RETRIEVE     rank the knowledge base (data/knowledge_base.json) against the request
                  with TF-IDF cosine similarity
  3. AUGMENT      turn the retrieved notes into a taste profile (genre, mood, target
                  energy, acoustic preference). The profile comes ONLY from retrieved
                  notes, so different retrieved notes produce different recommendations.
  4. RECOMMEND    score the song catalog with the existing recommender
  5. GENERATE     write an answer that cites the notes it used, with a confidence score.
                  The answer is built by a deterministic template, so the same request
                  always produces the same output and needs no network or API key.

Every stage is logged to logs/recommender.log.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.recommender import recommend_songs

logger = logging.getLogger("vibematch")

KB_PATH = "data/knowledge_base.json"
LOG_PATH = "logs/recommender.log"

MAX_QUERY_CHARS = 300
MIN_QUERY_CHARS = 3
MIN_SIMILARITY = 0.08   # a retrieved note below this similarity is ignored
TOP_K_DOCS = 3          # notes used to build the profile

# Confidence scoring (see compute_confidence)
STRONG_MATCH = 0.35        # top similarity at or above this counts as a fully strong match
CONFLICT_SPREAD = 0.40     # retrieved notes whose energy hints differ by more than this conflict
HIGH_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.45

STOPWORDS = frozenset(
    "a an and are as at be but by for from i im in is it me my of on or so some that "
    "the this to want with would like need something music songs song play give "
    "please can could just really very some something also too".split()
)

EXAMPLE_QUERIES = [
    "something calm for studying",
    "pump-up songs for the gym",
    "sad songs after a breakup",
    "music for a late night drive",
]


# --------------------------------------------------------------------------- logging

def setup_logging(log_path: str = LOG_PATH, level: int = logging.INFO) -> None:
    """Logs to a file (and warnings to the console). Safe to call more than once."""
    if logger.handlers:
        return
    logger.setLevel(level)
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError as exc:  # unwritable log dir must never break the app
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logger.addHandler(console)
        logger.warning("File logging disabled (%s)", exc)
    logger.propagate = False


# --------------------------------------------------------------------------- retrieval

def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase, split into words, drop stopwords, apply a light stemmer."""
    words = re.findall(r"[a-z]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    hint: Dict


def load_knowledge_base(path: str = KB_PATH) -> List[Doc]:
    """Loads and validates the knowledge base. Raises ValueError on a malformed file."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load knowledge base '{path}': {exc}") from exc

    docs = []
    for i, entry in enumerate(raw):
        try:
            docs.append(Doc(entry["id"], entry["title"], entry["text"], dict(entry["hint"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Knowledge base entry {i} is malformed: {exc!r}") from exc
    if not docs:
        raise ValueError(f"Knowledge base '{path}' is empty")
    return docs


class Retriever:
    """TF-IDF + cosine similarity over the knowledge base (pure Python, no extra deps)."""

    def __init__(self, docs: List[Doc]):
        self.docs = docs
        tokenized = [tokenize(d.title + " " + d.text) for d in docs]
        n = len(docs)
        df = Counter(tok for toks in tokenized for tok in set(toks))
        self.idf = {tok: math.log((1 + n) / (1 + count)) + 1 for tok, count in df.items()}
        self.vectors = [self._vectorize(toks) for toks in tokenized]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(t for t in tokens if t in self.idf)
        vec = {t: c * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def search(self, query: str, k: int = TOP_K_DOCS) -> List[Tuple[Doc, float]]:
        """Returns up to k (doc, similarity) pairs, best first, dropping near-zero matches."""
        q = self._vectorize(tokenize(query))
        if not q:
            return []
        scored = [(doc, sum(w * vec.get(t, 0.0) for t, w in q.items()))
                  for doc, vec in zip(self.docs, self.vectors)]
        scored = [(d, s) for d, s in scored if s >= MIN_SIMILARITY]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


# --------------------------------------------------------------------------- augmentation

def build_profile(retrieved: List[Tuple[Doc, float]], catalog_genres: set, catalog_moods: set) -> Dict:
    """
    Builds a taste profile (the dict score_song expects) from retrieved notes.
    Each note votes with weight = its similarity score. Genres/moods the catalog does
    not contain are ignored, so the profile can always be scored.
    """
    genre_votes: Dict[str, float] = defaultdict(float)
    mood_votes: Dict[str, float] = defaultdict(float)
    energy_sum = energy_w = acoustic_sum = acoustic_w = 0.0

    for doc, sim in retrieved:
        hint = doc.hint
        if hint.get("genre") in catalog_genres:
            genre_votes[hint["genre"]] += sim
        if hint.get("mood") in catalog_moods:
            mood_votes[hint["mood"]] += sim
        if isinstance(hint.get("energy"), (int, float)):
            energy_sum += sim * hint["energy"]
            energy_w += sim
        if isinstance(hint.get("likes_acoustic"), bool):
            acoustic_sum += sim * (1.0 if hint["likes_acoustic"] else 0.0)
            acoustic_w += sim

    profile: Dict = {}
    if genre_votes:
        profile["genre"] = max(genre_votes, key=genre_votes.get)
    if mood_votes:
        profile["mood"] = max(mood_votes, key=mood_votes.get)
    if energy_w:
        profile["energy"] = round(min(1.0, max(0.0, energy_sum / energy_w)), 2)
    if acoustic_w:
        profile["likes_acoustic"] = acoustic_sum / acoustic_w >= 0.5
    return profile


# --------------------------------------------------------------------------- confidence

def confidence_label(score: float) -> str:
    return "high" if score >= HIGH_CONFIDENCE else "low" if score < LOW_CONFIDENCE else "medium"


def compute_confidence(query: str, retrieved: List[Tuple[Doc, float]], retriever: "Retriever") -> Tuple[float, List[str]]:
    """
    Rates how sure the system is that its answer fits the request (0.0 to 1.0).

      strength  = top similarity / STRONG_MATCH, capped at 1
      coverage  = share of the request's content words that appear anywhere in the knowledge base
      confidence = strength * coverage, halved when the retrieved notes disagree about energy

    Returns (confidence, human-readable warnings).
    """
    warnings: List[str] = []
    if not retrieved:
        return 0.0, warnings

    strength = min(1.0, retrieved[0][1] / STRONG_MATCH)

    tokens = tokenize(query)
    unknown = sorted({t for t in tokens if t not in retriever.idf})
    coverage = 1 - len(unknown) / len(tokens) if tokens else 0.0
    if unknown:
        warnings.append("My notes don't cover: " + ", ".join(unknown) + ". Those words were ignored.")

    confidence = strength * coverage

    energies = [d.hint["energy"] for d, _ in retrieved if isinstance(d.hint.get("energy"), (int, float))]
    if len(energies) > 1 and max(energies) - min(energies) > CONFLICT_SPREAD:
        confidence *= 0.5
        warnings.append("Your request mixes very different energy levels, so I blended them; "
                        "the result may satisfy neither half.")

    return round(confidence, 2), warnings


# --------------------------------------------------------------------------- guardrails

def validate_query(query) -> Tuple[Optional[str], Optional[str]]:
    """Returns (clean_query, None) if usable, else (None, reason)."""
    if not isinstance(query, str):
        return None, "Request must be text."
    clean = " ".join(query.split())
    if len(clean) < MIN_QUERY_CHARS:
        return None, "Request is empty or too short."
    if len(clean) > MAX_QUERY_CHARS:
        return None, f"Request is too long (max {MAX_QUERY_CHARS} characters)."
    if not re.search(r"[A-Za-z]", clean):
        return None, "Request must contain words."
    return clean, None


# --------------------------------------------------------------------------- pipeline

@dataclass
class RagResult:
    status: str                       # "ok" | "invalid" | "no_match" | "error"
    query: str
    message: str = ""
    sources: List[Tuple[Doc, float]] = field(default_factory=list)
    profile: Dict = field(default_factory=dict)
    recommendations: List[Tuple[Dict, float, str]] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


def _template_answer(query: str, sources, profile: Dict, recs) -> str:
    lines = [f'You asked for: "{query}"', "", "What I looked up:"]
    for doc, sim in sources:
        lines.append(f"  - {doc.title} (match {sim:.2f}): {doc.text.split('. ')[0].rstrip('.')}.")
    parts = []
    if "genre" in profile:
        parts.append(f"genre {profile['genre']}")
    if "mood" in profile:
        parts.append(f"mood {profile['mood']}")
    if "energy" in profile:
        parts.append(f"energy near {profile['energy']:.2f}")
    if "likes_acoustic" in profile:
        parts.append("acoustic sounds" if profile["likes_acoustic"] else "electric/electronic sounds")
    lines += ["", "Based on those notes I looked for: " + ", ".join(parts) + ".", "", "My picks:"]
    for rank, (song, score, why) in enumerate(recs, start=1):
        lines.append(f"  {rank}. {song['title']} by {song['artist']} [{song['genre']}/{song['mood']}] - score {score:.2f}")
        lines.append(f"     {why}")
    return "\n".join(lines)


def ask(query, songs: List[Dict], retriever: Retriever, k: int = 5) -> RagResult:
    """Full RAG pipeline: request in, cited recommendations out. Never raises."""
    clean, problem = validate_query(query)
    if problem:
        logger.warning("Rejected request %r: %s", str(query)[:80], problem)
        return RagResult("invalid", str(query)[:80], message=problem)

    try:
        sources = retriever.search(clean)
        logger.info("Query %r retrieved %s", clean, [(d.id, round(s, 3)) for d, s in sources])
        if not sources:
            return RagResult(
                "no_match", clean,
                message="I couldn't find anything in my music notes for that. Try something like: "
                        + "; ".join(f'"{q}"' for q in EXAMPLE_QUERIES),
            )

        genres = {s["genre"] for s in songs}
        moods = {s["mood"] for s in songs}
        profile = build_profile(sources, genres, moods)
        if not profile:
            return RagResult("no_match", clean, message="The notes I found don't map onto this catalog.")

        confidence, warnings = compute_confidence(clean, sources, retriever)
        if confidence < LOW_CONFIDENCE:
            logger.warning("Low confidence %.2f for %r: %s", confidence, clean, warnings)
        recs = recommend_songs(profile, songs, k=k)
        logger.info("Profile %s -> top pick %s", profile, recs[0][0]["title"] if recs else None)

        answer = _template_answer(clean, sources, profile, recs)
        answer += f"\n\nConfidence: {confidence:.2f} ({confidence_label(confidence)})"
        if confidence < LOW_CONFIDENCE:
            answer += " - treat these picks with caution."
        for warning in warnings:
            answer += f"\nNote: {warning}"
        return RagResult("ok", clean, sources=sources, profile=profile, recommendations=recs,
                         answer=answer, confidence=confidence, warnings=warnings)
    except Exception as exc:  # last-resort guardrail
        logger.exception("Pipeline failure for %r", clean)
        return RagResult("error", clean, message=f"Something went wrong ({type(exc).__name__}); see {LOG_PATH}.")
