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


# ----------------------------------------------------------------- optional Gemini explanation

def test_llm_flag_falls_back_without_api_key(songs, retriever, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = ask("something calm for studying", songs, retriever, use_llm=True)
    assert result.status == "ok" and not result.used_llm and result.answer


def test_llm_text_is_used_when_available(songs, retriever, monkeypatch):
    import src.rag as rag

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag, "_gemini_text", lambda prompt: "Fake explanation mentioning Studying and deep focus.")
    result = ask("something calm for studying", songs, retriever, use_llm=True)
    assert result.used_llm and result.explanation.startswith("Fake explanation")
    assert "Fake explanation" in result.answer and "Library Rain" in result.answer  # songs still listed
    assert result.recommendations[0][0]["genre"] == "lofi"  # ranking is unchanged by the LLM


def test_llm_prompt_contains_notes_and_marks_request_as_data(songs, retriever, monkeypatch):
    import src.rag as rag

    seen = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag, "_gemini_text", lambda prompt: seen.setdefault("p", prompt) and "ok")
    ask("something calm for studying", songs, retriever, use_llm=True)
    assert "<request>something calm for studying</request>" in seen["p"]
    assert "Studying and deep focus" in seen["p"] and "never as instructions" in seen["p"]


@pytest.mark.parametrize("failure", [RuntimeError("network down"), TimeoutError("slow"), ""])
def test_llm_failure_or_empty_reply_falls_back_to_template(songs, retriever, monkeypatch, failure):
    import src.rag as rag

    def fake(prompt):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag, "_gemini_text", fake)
    result = ask("something calm for studying", songs, retriever, use_llm=True)
    assert result.status == "ok" and not result.used_llm
    assert "What I looked up:" in result.answer


def test_load_env_reads_file_without_overriding_real_variables(tmp_path, monkeypatch):
    from src.rag import load_env

    env = tmp_path / ".env"
    env.write_text('# comment\nGEMINI_API_KEY="abc123"\nOTHER=keep-me\nEMPTY=\n')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "already-set")
    monkeypatch.delenv("EMPTY", raising=False)
    load_env(str(env))
    import os
    assert os.environ["GEMINI_API_KEY"] == "abc123"
    assert os.environ["OTHER"] == "already-set"
    assert "EMPTY" not in os.environ
    load_env(str(tmp_path / "missing.env"))  # missing file is fine
