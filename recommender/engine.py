"""Hybrid recommendation engine.

Final score for every candidate resource =
      w_content       * ContentScore        (TF-IDF + cosine similarity, Rocchio feedback)
    + w_collaborative * CollaborativeScore  (item-item collaborative filtering)
    + w_knowledge     * KnowledgeScore      (rules: skill level fit + preferred format)
    + w_popularity    * PopularityScore     (global quality rating)

All signals are scaled to [0, 1]. Hard constraints (budget, max duration,
already-seen items, optional topic) are applied as filters before ranking.
The engine has no print/input calls so the same code can power a CLI or a UI.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_WEIGHTS, DISLIKE_THRESHOLD, LEVEL_INDEX, LEVELS, LIKE_THRESHOLD
from .profile import UserProfile

STRATEGIES = ["hybrid", "content", "collaborative", "knowledge", "popular"]
CF_SHRINKAGE = 0.5


@dataclass
class Recommendation:
    resource: dict
    score: float
    breakdown: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


class RecommenderEngine:
    def __init__(self, resources: pd.DataFrame, ratings: pd.DataFrame, weights: Optional[dict] = None):
        self.resources = resources.reset_index(drop=True)
        self.n = len(self.resources)
        self.id_to_idx = {int(rid): i for i, rid in enumerate(self.resources["id"])}
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._text = self._build_text()
        self._fit_content_model()
        self.set_ratings(ratings)

    # ------------------------------------------------------------------ models
    def _build_text(self) -> pd.Series:
        r = self.resources
        tags = r["tags"].str.replace(";", " ", regex=False)
        # topic and tags are repeated to give them more weight than free text
        return (r["title"] + " " + r["topic"] + " " + r["topic"] + " "
                + tags + " " + tags + " " + r["description"]).str.lower()

    def _fit_content_model(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(self._text)
        self.content_sim = cosine_similarity(self.tfidf)
        np.fill_diagonal(self.content_sim, 0.0)

    def set_ratings(self, ratings: pd.DataFrame) -> None:
        """(Re)build the item-item collaborative filtering model (adjusted cosine)."""
        self.ratings = ratings.copy()
        ratings = ratings[ratings["resource_id"].isin(self.id_to_idx)]
        if ratings.empty:
            self.cf_sim = np.zeros((self.n, self.n))
            return
        matrix = ratings.pivot_table(index="user_id", columns="resource_id", values="rating", aggfunc="mean")
        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)  # remove user bias
        centered = centered.reindex(columns=self.resources["id"], fill_value=0.0)
        self.cf_sim = cosine_similarity(centered.T.values)
        np.fill_diagonal(self.cf_sim, 0.0)

    def add_rating(self, user_id: str, resource_id: int, rating: float) -> pd.DataFrame:
        """Add/replace a user's rating in the community data and retrain CF."""
        df = self.ratings[~((self.ratings["user_id"] == user_id) & (self.ratings["resource_id"] == resource_id))]
        new_row = pd.DataFrame([{"user_id": user_id, "resource_id": int(resource_id), "rating": float(rating)}])
        self.set_ratings(pd.concat([df, new_row], ignore_index=True))
        return self.ratings

    # ----------------------------------------------------------------- signals
    def _query_vector(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text.lower()]).toarray()

    def content_scores(self, profile: UserProfile) -> np.ndarray:
        """Rocchio-style profile vector: interests/goals + liked items - disliked items."""
        query_text = " ".join(profile.interests * 2) + " " + profile.goals
        q = self._query_vector(query_text)
        has_query = q.any()
        liked = self._idx_where(profile, lambda r: r >= LIKE_THRESHOLD)
        disliked = self._idx_where(profile, lambda r: r <= DISLIKE_THRESHOLD)

        vec = q if has_query else np.zeros_like(q)
        if liked:
            liked_vec = np.asarray(self.tfidf[liked].mean(axis=0))
            vec = 0.6 * vec + 0.4 * liked_vec if has_query else liked_vec
        if disliked:
            vec = vec - 0.25 * np.asarray(self.tfidf[disliked].mean(axis=0))
        vec = np.clip(vec, 0, None)
        if not vec.any():
            return np.zeros(self.n)
        sims = cosine_similarity(vec, self.tfidf).ravel()
        return sims / sims.max() if sims.max() > 0 else sims

    def collaborative_scores(self, profile: UserProfile) -> Optional[np.ndarray]:
        """Predict preference from similarity to items the user rated (neutral = 3)."""
        rated = [(self.id_to_idx[rid], r) for rid, r in profile.ratings.items() if rid in self.id_to_idx]
        if not rated:
            return None
        idxs = [i for i, _ in rated]
        dev = np.array([r - 3.0 for _, r in rated])       # -2 .. +2
        sims = self.cf_sim[:, idxs]
        num = sims @ dev
        # shrinkage term (+CF_SHRINKAGE) keeps predictions near neutral when evidence is weak
        den = np.abs(sims).sum(axis=1) + CF_SHRINKAGE
        pred = num / den
        return np.clip((pred + 2.0) / 4.0, 0, 1)         # map to 0..1 (0.5 = neutral)

    def knowledge_scores(self, profile: UserProfile) -> np.ndarray:
        user_lvl = LEVEL_INDEX[profile.level]
        gaps = self.resources["level"].map(LEVEL_INDEX).fillna(0).astype(int).values - user_lvl
        level_fit = np.select([gaps == 0, gaps == 1, gaps == -1], [1.0, 0.75, 0.45], default=0.15)
        if profile.preferred_formats:
            fmt_fit = self.resources["format"].isin(profile.preferred_formats).astype(float).values
            fmt_fit = np.where(fmt_fit > 0, 1.0, 0.4)
        else:
            fmt_fit = np.ones(self.n)
        return 0.7 * level_fit + 0.3 * fmt_fit

    def popularity_scores(self) -> np.ndarray:
        return np.clip((self.resources["rating"].values - 3.0) / 2.0, 0, 1)

    # ----------------------------------------------------------------- helpers
    def _idx_where(self, profile: UserProfile, cond) -> list:
        return [self.id_to_idx[rid] for rid, r in profile.ratings.items() if rid in self.id_to_idx and cond(r)]

    def _filter_mask(self, profile: UserProfile, topic: Optional[str], include_seen: bool,
                     apply_constraints: bool = True) -> np.ndarray:
        r = self.resources
        mask = np.ones(self.n, dtype=bool)
        if apply_constraints:
            if profile.free_only:
                mask &= (r["cost"].str.lower() == "free").values
            if profile.max_duration:
                mask &= (r["duration_hours"] <= profile.max_duration).values
        if topic:
            mask &= (r["topic"].str.lower() == topic.lower()).values
        if not include_seen:
            seen = set(profile.ratings) | set(profile.completed)
            mask &= ~r["id"].isin(seen).values
        return mask

    def get_resource(self, resource_id: int) -> Optional[dict]:
        idx = self.id_to_idx.get(int(resource_id))
        return None if idx is None else self.resources.iloc[idx].to_dict()

    def community_stats(self, resource_id: int) -> tuple:
        r = self.ratings[self.ratings["resource_id"] == int(resource_id)]["rating"]
        return (round(r.mean(), 2) if len(r) else None, len(r))

    # --------------------------------------------------------- recommendations
    def recommend(self, profile: UserProfile, top_n: int = 10, strategy: str = "hybrid",
                  topic: Optional[str] = None, include_seen: bool = False,
                  weights: Optional[dict] = None) -> list:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")

        content = self.content_scores(profile)
        cf = self.collaborative_scores(profile)
        knowledge = self.knowledge_scores(profile)
        popularity = self.popularity_scores()
        signals = {"content": content, "collaborative": cf, "knowledge": knowledge, "popularity": popularity}

        if strategy == "hybrid":
            w = dict(weights or self.weights)
            if cf is None:                      # cold start: no ratings yet
                w["collaborative"] = 0.0
            if not content.any():               # no interests given
                w["content"] = 0.0
            total = sum(w.values()) or 1.0
            w = {k: v / total for k, v in w.items()}
            final = sum(w[k] * (signals[k] if signals[k] is not None else 0) for k in w)
        elif strategy == "collaborative":
            final = cf if cf is not None else popularity   # fallback for cold start
        elif strategy == "popular":
            final = popularity
        else:
            final = signals[strategy]

        mask = self._filter_mask(profile, topic, include_seen)
        order = [i for i in np.argsort(-final, kind="stable") if mask[i]][:top_n]

        results = []
        for i in order:
            breakdown = {k: (round(float(v[i]), 3) if v is not None else None) for k, v in signals.items()}
            results.append(Recommendation(
                resource=self.resources.iloc[i].to_dict(),
                score=round(float(final[i]), 4),
                breakdown=breakdown,
                reasons=self._explain(profile, i, breakdown),
            ))
        return results

    def _explain(self, profile: UserProfile, i: int, b: dict) -> list:
        """Human readable explanation of why an item was recommended (Explainable AI)."""
        item = self.resources.iloc[i]
        text = self._text.iloc[i]
        reasons = []

        matched = [kw for kw in profile.interests if kw.lower() in text]
        if matched:
            reasons.append("Matches your interest in " + ", ".join(matched[:3]))
        elif b["content"] and b["content"] >= 0.5:
            reasons.append("Closely related to your interests/goals")

        liked = self._idx_where(profile, lambda r: r >= LIKE_THRESHOLD)
        if liked:
            best = max(liked, key=lambda j: self.content_sim[i, j] + self.cf_sim[i, j])
            if self.content_sim[i, best] + self.cf_sim[i, best] > 0.2:
                title = self.resources.iloc[best]["title"]
                reasons.append(f"Similar to '{title}' which you liked")
        if b["collaborative"] is not None and b["collaborative"] >= 0.65:
            reasons.append("Learners with similar taste rated it highly")

        gap = LEVEL_INDEX.get(item["level"], 0) - LEVEL_INDEX[profile.level]
        reasons.append({0: f"Right at your {profile.level} level",
                        1: "One step up - good to challenge yourself",
                        -1: "Slightly below your level - good for revision"}.get(gap, f"{item['level']} level"))

        if profile.preferred_formats and item["format"] in profile.preferred_formats:
            reasons.append(f"In your preferred format ({item['format']})")
        if item["rating"] >= 4.7:
            reasons.append(f"Highly rated ({item['rating']}/5)")
        if item["cost"].lower() == "free":
            reasons.append("Free")
        return reasons

    # ------------------------------------------------------------ other tools
    def search(self, query: str, top_n: int = 10, profile: Optional[UserProfile] = None) -> list:
        q = self._query_vector(query)
        if not q.any():
            return []
        sims = cosine_similarity(q, self.tfidf).ravel()
        mask = self._filter_mask(profile, None, True) if profile else np.ones(self.n, bool)
        order = [i for i in np.argsort(-sims) if sims[i] > 0 and mask[i]][:top_n]
        return [(self.resources.iloc[i].to_dict(), round(float(sims[i]), 3)) for i in order]

    def similar_resources(self, resource_id: int, top_n: int = 5) -> list:
        idx = self.id_to_idx.get(int(resource_id))
        if idx is None:
            return []
        cf = np.clip(self.cf_sim[idx], 0, None)
        score = 0.7 * self.content_sim[idx] + 0.3 * cf
        order = [i for i in np.argsort(-score) if i != idx][:top_n]
        return [(self.resources.iloc[i].to_dict(), round(float(score[i]), 3)) for i in order]

    def learning_path(self, profile: UserProfile, goal: str, per_level: int = 2) -> list:
        """Build an ordered path from the user's level up to Advanced for a goal/topic."""
        q = self._query_vector(goal)
        if not q.any():
            return []
        rel = cosine_similarity(q, self.tfidf).ravel()
        pop = self.popularity_scores()
        kfit = self.knowledge_scores(profile)
        mask = self._filter_mask(profile, None, include_seen=False) & (rel > 0.05)
        score = 0.65 * rel + 0.2 * pop + 0.15 * kfit

        stages = []
        for level in LEVELS[LEVEL_INDEX[profile.level]:]:
            idxs = [i for i in np.argsort(-score) if mask[i] and self.resources.iloc[i]["level"] == level]
            items = [self.resources.iloc[i].to_dict() for i in idxs[:per_level]]
            if items:
                stages.append({"level": level, "resources": items,
                               "hours": sum(x["duration_hours"] for x in items)})
        return stages
