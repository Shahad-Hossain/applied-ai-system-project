"""
VibeMatch 2.0: Streamlit interface for the retrieval-augmented music recommender.

Run from the project root with:  streamlit run app.py
"""

import streamlit as st

from src.rag import (
    EXAMPLE_QUERIES, MAX_QUERY_CHARS, LOW_CONFIDENCE, Retriever, ask, confidence_label,
    llm_available, load_env, load_knowledge_base, setup_logging,
)
from src.recommender import load_songs

st.set_page_config(page_title="VibeMatch 2.0", page_icon="🎵", layout="centered")


@st.cache_resource
def get_system():
    """Loads the catalog and knowledge base once per server run."""
    load_env()
    setup_logging()
    return load_songs("data/songs.csv"), Retriever(load_knowledge_base())


def render_result(result) -> None:
    if result.status != "ok":
        (st.warning if result.status in ("invalid", "no_match") else st.error)(result.message)
        return

    label = confidence_label(result.confidence)
    col1, col2 = st.columns([1, 3])
    col1.metric("Confidence", f"{result.confidence:.2f}", label)
    col2.progress(min(1.0, result.confidence), text=f"{label} confidence")
    if result.confidence < LOW_CONFIDENCE:
        st.warning("Low confidence: treat these picks with caution.")
    for warning in result.warnings:
        st.info(warning)

    if result.used_llm:
        st.subheader("Why these songs (written by Gemini)")
        st.write(result.explanation)

    st.subheader("Recommended songs")
    for rank, (song, score, why) in enumerate(result.recommendations, start=1):
        with st.expander(f"{rank}. {song['title']} by {song['artist']}   ·   {song['genre']} / {song['mood']}   ·   {score:.2f} / 5", expanded=rank <= 2):
            st.progress(min(1.0, score / 5.0))
            for reason in why.split(", "):
                st.write(f"- {reason}")

    with st.expander("What I looked up (retrieved notes)", expanded=not result.used_llm):
        for doc, sim in result.sources:
            st.markdown(f"**{doc.title}** (match {sim:.2f})")
            st.caption(doc.text)

    with st.expander("Taste profile built from those notes"):
        st.json(result.profile)


def main() -> None:
    songs, retriever = get_system()

    st.title("🎵 VibeMatch 2.0")
    st.caption("Describe what you want to hear. VibeMatch retrieves matching music notes, builds a taste profile "
               "from them, and ranks a 20-song catalog with a score breakdown for every pick.")

    if "query" not in st.session_state:
        st.session_state.query = ""

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, example in zip(cols, EXAMPLE_QUERIES):
        if col.button(example, use_container_width=True):
            st.session_state.query = example

    with st.sidebar:
        st.header("Options")
        has_key = llm_available()
        use_llm = st.checkbox("Write the explanation with Gemini", value=False, disabled=not has_key)
        if not has_key:
            st.caption("To enable this, paste your key into the `.env` file as `GEMINI_API_KEY=...` "
                       "and restart the app.")
        else:
            st.caption("Optional. If Gemini fails, the built-in template is used instead.")
        st.divider()
        st.caption("The catalog is 20 invented songs, and the knowledge base is 26 hand-written notes. "
                   "Results are only as good as those notes.")

    query = st.text_input("What do you want to listen to?", key="query", max_chars=MAX_QUERY_CHARS,
                          placeholder="e.g. something calm for studying")
    if st.button("Recommend", type="primary") or query:
        result = ask(query, songs, retriever, use_llm=use_llm)
        if use_llm and result.status == "ok" and not result.used_llm:
            st.info("Gemini was unavailable, so the built-in template was used. Details are in logs/recommender.log.")
        render_result(result)


main()
