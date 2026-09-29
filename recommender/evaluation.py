"""Offline evaluation of the recommender (hold-out Precision@K / Recall@K / Hit-rate).

For every community learner with enough liked items, a fraction of their liked
items is hidden. A profile is rebuilt from the remaining ratings (interests =
their most-liked topics, level = most common level of liked items) and each
strategy is asked for K recommendations. We then check how many hidden items
were recovered. A random recommender is included as a baseline.
"""

from collections import Counter

import numpy as np
import pandas as pd

from .config import LIKE_THRESHOLD
from .engine import STRATEGIES, RecommenderEngine
from .profile import UserProfile


def _split(ratings: pd.DataFrame, test_frac: float, rng: np.random.Generator, min_liked: int = 4):
    train_parts, test = [], {}
    for uid, grp in ratings.groupby("user_id"):
        liked = grp[grp["rating"] >= LIKE_THRESHOLD]
        if len(liked) < min_liked:
            train_parts.append(grp)
            continue
        n_test = max(1, int(round(len(liked) * test_frac)))
        test_ids = set(rng.choice(liked["resource_id"].values, size=n_test, replace=False).tolist())
        test[uid] = test_ids
        train_parts.append(grp[~grp["resource_id"].isin(test_ids)])
    return pd.concat(train_parts, ignore_index=True), test


def _profile_from_history(uid: str, train: pd.DataFrame, resources: pd.DataFrame) -> UserProfile:
    user = train[train["user_id"] == uid].merge(resources, left_on="resource_id", right_on="id")
    liked = user[user["rating_x"] >= LIKE_THRESHOLD]
    topics = [t for t, _ in Counter(liked["topic"]).most_common(2)]
    level = Counter(liked["level"]).most_common(1)[0][0] if len(liked) else "Beginner"
    ratings = dict(zip(user["resource_id"].astype(int), user["rating_x"].astype(float)))
    return UserProfile(username=uid, level=level, interests=topics, ratings=ratings)


def evaluate(resources: pd.DataFrame, ratings: pd.DataFrame, k: int = 5,
             test_frac: float = 0.3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train, test = _split(ratings, test_frac, rng)
    engine = RecommenderEngine(resources, train)
    all_ids = resources["id"].tolist()

    metrics = {s: {"precision": [], "recall": [], "hit": [], "items": set()} for s in STRATEGIES + ["random"]}
    for uid, hidden in test.items():
        profile = _profile_from_history(uid, train, resources)
        seen = set(profile.ratings)
        for strategy in metrics:
            if strategy == "random":
                pool = [i for i in all_ids if i not in seen]
                rec_ids = rng.choice(pool, size=k, replace=False).tolist()
            else:
                rec_ids = [int(r.resource["id"]) for r in engine.recommend(profile, top_n=k, strategy=strategy)]
            hits = len(set(rec_ids) & hidden)
            m = metrics[strategy]
            m["precision"].append(hits / k)
            m["recall"].append(hits / len(hidden))
            m["hit"].append(1.0 if hits else 0.0)
            m["items"].update(rec_ids)

    rows = []
    for strategy, m in metrics.items():
        rows.append({
            "strategy": strategy,
            f"precision@{k}": round(float(np.mean(m["precision"])), 3),
            f"recall@{k}": round(float(np.mean(m["recall"])), 3),
            f"hit_rate@{k}": round(float(np.mean(m["hit"])), 3),
            "coverage": round(len(m["items"]) / len(all_ids), 3),
        })
    df = pd.DataFrame(rows).sort_values(f"precision@{k}", ascending=False).reset_index(drop=True)
    df.attrs["users_evaluated"] = len(test)
    return df
