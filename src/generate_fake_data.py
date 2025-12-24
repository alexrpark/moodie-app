from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from datetime import datetime, timedelta
from pathlib import Path
import random

from storage import append_row_csv

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "mood_log_fake.csv"

LOW_NOTES = [
    "anxious about work",
    "slept badly, tired",
    "overwhelmed and distracted",
    "argument, feeling low",
    "headache and stressed",
]
MID_NOTES = [
    "normal day, steady",
    "busy but okay",
    "a bit tired but fine",
    "productive, nothing special",
    "routine day",
]
HIGH_NOTES = [
    "great workout, energized",
    "good time with friends",
    "felt calm and focused",
    "excited, made progress",
    "slept well, feeling great",
]

def pick_note(mood: int) -> str:
    if mood <= 1:
        return random.choice(LOW_NOTES)
    if mood == 2 or mood == 3:
        return random.choice(MID_NOTES)
    return random.choice(HIGH_NOTES)

def main() -> None:
    random.seed(42)  # reproducible output

    days = 45
    entries_per_day = (1, 3)  # min/max entries per day
    start_date = datetime.now() - timedelta(days=days)

    # Create some structure: Mondays a bit lower, weekends a bit higher
    for d in range(days):
        day_dt = start_date + timedelta(days=d)
        weekday = day_dt.weekday()  # Mon=0 ... Sun=6

        n = random.randint(entries_per_day[0], entries_per_day[1])
        for i in range(n):
            # base mood around 3 with noise
            base = 3 + random.choice([-1, 0, 0, 1])

            if weekday == 0:  # Monday dip
                base -= 1
            if weekday in (5, 6):  # weekend lift
                base += 1

            mood = max(0, min(5, base))
            note = pick_note(mood)

            # spread entries through the day a bit
            ts = (day_dt + timedelta(hours=9 + i * 4)).isoformat(timespec="seconds")

            row = {
                "timestamp": ts,
                "mood": mood,
                "note": note,
                "sentiment": float(analyzer.polarity_scores(note)["compound"]),
            }
            append_row_csv(OUT_PATH, row)

    print(f"Wrote fake data to: {OUT_PATH}")

if __name__ == "__main__":
    main()
