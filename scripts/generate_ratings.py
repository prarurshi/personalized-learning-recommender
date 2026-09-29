"""Generate a synthetic community ratings dataset (data/ratings.csv).

Each synthetic learner has 1-2 favourite topics and a skill level. They are
more likely to rate (and rate highly) resources in their favourite topics that
match their level. This gives the collaborative-filtering model realistic
patterns to learn from. Run:  python scripts/generate_ratings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources.csv"
OUTPUT = ROOT / "data" / "ratings.csv"

LEVELS = ["Beginner", "Intermediate", "Advanced"]
N_USERS = 60
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = pd.read_csv(RESOURCES)
    topics = sorted(res["topic"].unique())
    rows = []

    for u in range(1, N_USERS + 1):
        user_id = f"learner_{u:03d}"
        favs = set(rng.choice(topics, size=rng.integers(1, 3), replace=False))
        level = rng.integers(0, 3)

        for _, item in res.iterrows():
            item_level = LEVELS.index(item["level"])
            level_gap = abs(item_level - level)
            quality = item["rating"] - 4.5  # small bias from global quality

            if item["topic"] in favs:
                if rng.random() > 0.65:
                    continue
                score = rng.normal(4.4, 0.5) + quality - 0.6 * level_gap
            else:
                if rng.random() > 0.07:
                    continue
                score = rng.normal(2.9, 0.9) + quality

            rows.append((user_id, int(item["id"]), int(np.clip(round(score), 1, 5))))

    df = pd.DataFrame(rows, columns=["user_id", "resource_id", "rating"])
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} ratings from {df.user_id.nunique()} learners -> {OUTPUT}")


if __name__ == "__main__":
    main()
