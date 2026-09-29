"""Loading and saving of the resource catalogue and community ratings."""

from pathlib import Path

import pandas as pd

from .config import RATINGS_FILE, RESOURCES_FILE

REQUIRED_COLUMNS = [
    "id", "title", "topic", "level", "format", "duration_hours",
    "cost", "rating", "provider", "tags", "description",
]


def load_resources(path: Path = RESOURCES_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"resources file is missing columns: {sorted(missing)}")
    df["id"] = df["id"].astype(int)
    df["duration_hours"] = df["duration_hours"].astype(float)
    df["rating"] = df["rating"].astype(float)
    for col in ["title", "topic", "level", "format", "cost", "provider", "tags", "description"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def load_ratings(path: Path = RATINGS_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["user_id", "resource_id", "rating"])
    df = pd.read_csv(path)
    df["user_id"] = df["user_id"].astype(str)
    df["resource_id"] = df["resource_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df


def save_ratings(df: pd.DataFrame, path: Path = RATINGS_FILE) -> None:
    out = df.copy()
    out["rating"] = out["rating"].round().astype(int)
    out.to_csv(path, index=False)
