from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# IMPORTANT: so "from storage import ..." works when running from web/
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from storage import append_row_csv  # noqa: E402


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mood_log.csv"
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.secret_key = "dev-only-change-me"  # fine for local dev


def sentiment_compound(text: str) -> float:
    if not text.strip():
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


@app.get("/")
def home():
    return redirect(url_for("log"))


@app.get("/log")
@app.post("/log")
def log():
    if request.method == "POST":
        mood_raw = request.form.get("mood", "").strip()
        note = (request.form.get("note") or "").strip()
        ts_raw = (request.form.get("timestamp") or "").strip()

        try:
            mood = int(mood_raw)
        except ValueError:
            flash("Mood must be a whole number 0–5.")
            return redirect(url_for("log"))

        if not (0 <= mood <= 5):
            flash("Mood must be between 0 and 5.")
            return redirect(url_for("log"))
        
        if ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                flash("Invalid timestamp format.")
                return redirect(url_for("log"))
        else:
            ts = datetime.now()

        row = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "mood": mood,
            "note": note,
            "sentiment": sentiment_compound(note),
        }
        append_row_csv(DATA_PATH, row)
        flash("Saved!")
        return redirect(url_for("log"))

    default_ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    return render_template("log.html", title="Log", default_ts=default_ts)


@app.get("/trends")
def trends():
    if not DATA_PATH.exists():
        return render_template("trends.html", title="Trends", found=False)

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return render_template("trends.html", title="Trends", found=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date", as_index=False).agg(avg_mood=("mood", "mean"))
    last_7 = daily.tail(7)
    prev_7 = daily.iloc[-14:-7]
    chart_labels = [str(d) for d in last_7["date"].tolist()]
    chart_values = [round(float(x), 2) for x in last_7["avg_mood"].tolist()]

    avg_all = float(df["mood"].mean())
    avg_7d = float(last_7["avg_mood"].mean())
    high_7d = float(last_7["avg_mood"].max())
    low_7d = float(last_7["avg_mood"].min())
    std_7d = float(last_7["avg_mood"].std()) if len(last_7) >=2 else None
    std_prev7 = float(prev_7["avg_mood"].std()) if len(prev_7) >=2 else None

    volatility_msg = None
    if std_7d is None or std_prev7 is None:
        volatility_msg = "Not enough data to assess mood volatility (requires 2 weeks)."
    else:
        if std_7d > std_prev7 + 0.2:
            volatility_msg = "Your mood was more volatile this week than last week."
        elif std_7d < std_prev7 - 0.2:
            volatility_msg = "Your mood was steadier this week than last week."
        else:
            volatility_msg = "Your mood stability was similar to last week."

    return render_template(
        "trends.html",
        title="Trends",
        found=True,
        total_entries=len(df),
        avg_all=f"{avg_all:.2f}",
        avg_7d=f"{avg_7d:.2f}",
        high_7d=f"{high_7d:.2f}",
        low_7d=f"{low_7d:.2f}",
        volatility_msg=volatility_msg,
        chart_labels=chart_labels,
        chart_values=chart_values,
    )


if __name__ == "__main__":
    app.run(debug=True)

