from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

DEFAULT_COLUMNS: List[str] = ["timestamp", "mood", "note", "sentiment"]

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def append_row_csv(csv_path: Path, row: Dict[str, object], columns: List[str] = DEFAULT_COLUMNS) -> None:
    """Append a row to a CSV, creating it with a header if it doesn't exist."""
    ensure_parent_dir(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        filtered = {col: row.get(col, "") for col in columns}
        writer.writerow(filtered)
