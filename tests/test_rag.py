import pytest

from src.rag import (
    MAX_QUERY_CHARS, Retriever, ask, build_profile, load_knowledge_base, tokenize, validate_query,
)
from src.recommender import load_songs


@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_knowledge_base())


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("I want songs for studying") == ["study"]


def test_retrieval_finds_relevant_note(retriever):
    docs = [d.id for d, _ in retriever.search("pump-up songs for the gym")]
    assert docs[0] == "act_workout"


def test_retrieval_returns_nothing_for_gibberish(retriever):
    assert retriever.search("asdfgh qwerty") == []


def test_retrieved_notes_change_recommendations(songs, retriever):
    study = ask("something calm for studying", songs, retriever)
    gym = ask("pump-up songs for the gym", songs, retriever)
    assert study.status == gym.status == "ok"
    assert study.profile["energy"] < 0.5 < gym.profile["energy"]
    assert study.recommendations[0][0]["title"] != gym.recommendations[0][0]["title"]
    assert study.recommendations[0][0]["genre"] == "lofi"
    assert gym.recommendations[0][0]["genre"] == "edm"


def test_answer_cites_retrieved_notes(songs, retriever):
    result = ask("sad songs after a breakup", songs, retriever)
    assert result.sources
    for doc, _ in result.sources:
        assert doc.title in result.answer


def test_same_query_is_deterministic(songs, retriever):
    a = ask("music for a late night drive", songs, retriever)
    b = ask("music for a late night drive", songs, retriever)
    assert [s["id"] for s, _, _ in a.recommendations] == [s["id"] for s, _, _ in b.recommendations]


def test_profile_ignores_genres_missing_from_catalog(retriever):
    sources = retriever.search("workout at the gym")
    profile = build_profile(sources, catalog_genres={"pop"}, catalog_moods={"happy"})
    assert "genre" not in profile or profile["genre"] == "pop"


@pytest.mark.parametrize("bad", [None, 42, "", "   ", "ab", "12345", "x" * (MAX_QUERY_CHARS + 1)])
def test_guardrail_rejects_bad_input(bad, songs, retriever):
    assert validate_query(bad)[0] is None
    result = ask(bad, songs, retriever)
    assert result.status == "invalid" and result.message


def test_unmatched_request_fails_safely(songs, retriever):
    result = ask("qwerty zxcvb", songs, retriever)
    assert result.status == "no_match" and result.recommendations == []


def test_bad_knowledge_base_raises_clear_error(tmp_path):
    bad = tmp_path / "kb.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        load_knowledge_base(str(bad))


# ----------------------------------------------------------------- confidence scoring

def test_confidence_high_for_clear_request(songs, retriever):
    result = ask("something calm for studying", songs, retriever)
    assert result.confidence >= 0.7 and "Confidence:" in result.answer


def test_confidence_low_when_words_are_not_covered(songs, retriever):
    clear = ask("something calm for studying", songs, retriever)
    vague = ask("classical piano", songs, retriever)
    assert vague.confidence < 0.45 < clear.confidence
    assert any("classical" in w for w in vague.warnings)
    assert "caution" in vague.answer


def test_conflicting_energy_is_flagged_and_lowers_confidence(songs, retriever):
    result = ask("something sad but also really loud", songs, retriever)
    assert any("mixes very different energy" in w for w in result.warnings)
    assert result.confidence <= 0.5


# ----------------------------------------------------------------- evaluation harness

def test_evaluation_suite_has_no_unexpected_failures(songs, retriever):
    from src.evaluate import evaluate

    rows, repeat_ok, repeat_total = evaluate(songs, retriever)
    failures = [c["query"] for c, ok, _, _ in rows if not ok and not c.get("known_limit")]
    assert failures == []
    assert repeat_ok == repeat_total
