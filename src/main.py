"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

Run from the project root with: python -m src.main
Options:
  --profile NAME      run a single profile instead of all of them
  --experiment NAME   weight_shift (energy x2, genre x0.5) or no_mood (mood check off)
  --ask "TEXT"        free-text request answered with retrieval-augmented recommendations
  --llm               with --ask, let Claude write the answer (needs ANTHROPIC_API_KEY)
"""

import argparse

from src.rag import Retriever, ask, load_knowledge_base, setup_logging
from src.recommender import DEFAULT_WEIGHTS, load_songs, recommend_songs


# Taste profiles: target values for the features used in scoring.
# "energy" is a 0.0-1.0 target (closer songs score higher), not a minimum.
PROFILES = {
    # Normal, well-behaved users
    "pop_happy": {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False},
    "chill_lofi": {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
    "intense_rock": {"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
    # Adversarial / edge-case users
    "conflicted_sad_but_loud": {"genre": "folk", "mood": "melancholy", "energy": 0.9, "likes_acoustic": True},
    "mellow_rock": {"genre": "rock", "mood": "chill", "energy": 0.40, "likes_acoustic": True},
    "unknown_tastes": {"genre": "classical", "mood": "sad", "energy": 0.5, "likes_acoustic": False},
}
DEFAULT_PROFILE = "pop_happy"

# Experimental weight sets (default recipe totals 5.0 points)
EXPERIMENTS = {
    "weight_shift": {**DEFAULT_WEIGHTS, "energy": DEFAULT_WEIGHTS["energy"] * 2, "genre": DEFAULT_WEIGHTS["genre"] / 2},
    "no_mood": {**DEFAULT_WEIGHTS, "mood": 0.0},
}


def show(name: str, user_prefs: dict, songs: list, weights: dict | None = None) -> None:
    """Prints the top 5 recommendations and reasons for one profile."""
    max_score = sum((weights or DEFAULT_WEIGHTS).values())
    print(f"\nProfile: {name} {user_prefs}")
    print("Top recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommend_songs(user_prefs, songs, k=5, weights=weights), start=1):
        print(f"{rank}. {song['title']} by {song['artist']} [{song['genre']}/{song['mood']}]")
        print(f"   Score: {score:.2f} / {max_score:.2f}")
        for reason in explanation.split(", "):
            print(f"   - {reason}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--ask", metavar="TEXT", help="free-text request (RAG mode)")
    parser.add_argument("--llm", action="store_true", help="use Claude to write the --ask answer")
    args = parser.parse_args()

    setup_logging()
    try:
        songs = load_songs("data/songs.csv")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Could not load data/songs.csv: {exc}")
    print(f"Loaded songs: {len(songs)}")

    if args.ask is not None:
        try:
            retriever = Retriever(load_knowledge_base())
        except ValueError as exc:
            raise SystemExit(str(exc))
        result = ask(args.ask, songs, retriever, use_llm=args.llm)
        print()
        print(result.answer if result.status == "ok" else f"[{result.status}] {result.message}")
        if args.llm and result.status == "ok" and not result.used_llm:
            print("\n(Claude was unavailable, so the built-in template wrote this answer.)")
        return

    weights = EXPERIMENTS[args.experiment] if args.experiment else None
    if weights:
        print(f"Experiment: {args.experiment} weights = {weights}")

    names = [args.profile] if args.profile else list(PROFILES)
    for name in names:
        show(name, PROFILES[name], songs, weights)


if __name__ == "__main__":
    main()
