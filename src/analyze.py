from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse
import re

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mood_log.csv"

STOPWORDS = {"but", "day", "okay", "normal", "routine", "steady", "productive", "busy"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze mood log CSV and print insights.")
    p.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to mood log CSV (default: data/mood_log.csv)",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    data_path = Path(args.file)

    if not data_path.exists():
        print(f"No data at: {data_path}")
        return

    df = pd.read_csv(data_path)
    if df.empty:
        print("No rows yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    df["date"] = df["timestamp"].dt.date
    df_daily = df.groupby("date", as_index=False).agg(
        avg_mood=("mood", "mean"),
        notes=("note", lambda x: " ".join(x.dropna().astype(str))),
    )

    # Daily average sentiment (same "daily" level as avg_mood)
    df_sent = df.groupby("date", as_index=False).agg(avg_sent=("sentiment", "mean"))
    df_daily2 = df_daily.merge(df_sent, on="date", how="left")

    # Correlation between daily mood and daily sentiment
    if df_daily2["avg_sent"].notna().sum() >= 10:
        corr = df_daily2["avg_mood"].corr(df_daily2["avg_sent"])
        if pd.notna(corr):
            print(f"\nMood ↔ sentiment correlation: {corr:.2f}")

    print("=== Summary ===")
    print(f"File: {data_path}")
    print(f"Days tracked: {len(df_daily)}")
    print(f"Total entries: {len(df)}")

    # ---- Signal 1: 7-day trend ----
    if len(df_daily) >= 10:
        last7 = df_daily.tail(7)["avg_mood"].mean()
        prev7 = df_daily.iloc[-14:-7]["avg_mood"].mean()
        delta = last7 - prev7
        direction = "up" if delta > 0.15 else "down" if delta < -0.15 else "flat"
        print(f"\nTrend (7d vs prior 7d): {direction} ({delta:+.2f})")
    else:
        print("\nTrend: not enough data yet")

    # ---- Signal 3: weekday bias (only report if 2+ weekdays qualify) ----
    df["weekday"] = df["timestamp"].dt.day_name()
    weekday_stats = df.groupby("weekday").agg(avg=("mood", "mean"), count=("mood", "count"))
    weekday_stats = weekday_stats[weekday_stats["count"] >= 3]

    if len(weekday_stats) >= 2:
        worst = weekday_stats.sort_values("avg").iloc[0]
        best = weekday_stats.sort_values("avg").iloc[-1]
        print(f"\nBest weekday: {best.name} ({best['avg']:.2f})")
        print(f"Worst weekday: {worst.name} ({worst['avg']:.2f})")
    else:
        print("\nWeekday pattern: not enough data yet")

    # ---- Signal 4: words on low days ----
    if len(df_daily) >= 10:
        low_cut = df_daily["avg_mood"].quantile(0.3)
        low_days = df_daily[df_daily["avg_mood"] <= low_cut]
        tokens = []
        for t in low_days["notes"].astype(str):
            tokens.extend([w for w in re.findall(r"[a-z]{2,}", t.lower()) if w not in STOPWORDS])

        common = [w for w, _ in Counter(tokens).most_common(8)]
        if common:
            print("\nWords common on lower-mood days:", ", ".join(common))


if __name__ == "__main__":
    main()