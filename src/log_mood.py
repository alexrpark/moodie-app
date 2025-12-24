from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from storage import append_row_csv

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mood_log.csv"
analyzer = SentimentIntensityAnalyzer()

def prompt_mood() -> int:
    while True:
        raw = input("how are you feeling today alex (0-5)? ").strip()
        try:
            val = int(raw)
        except ValueError:
            print("sorry! it has to be between 0 and 5.")
            continue
        if 0 <= val <= 5:
            return val
        print("mood must be between 0 and 5.")

def prompt_note() -> str:
    return input("add a note (optional): ").strip()

def sentiment_compound(text: str) -> float:
    if not text.strip():
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])

def main() -> None:
    mood = prompt_mood()
    note = prompt_note()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mood": mood,
        "note": note,
        "sentiment": sentiment_compound(note),
    }

    append_row_csv(DATA_PATH, row)
    print(f"saved entry to {DATA_PATH}")

if __name__ == "__main__":
    main()
