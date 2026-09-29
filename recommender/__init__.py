"""Personalized Learning Resource Recommender - core package (UI independent)."""

from .config import FORMATS, LEVELS
from .data_loader import load_ratings, load_resources, save_ratings
from .engine import STRATEGIES, Recommendation, RecommenderEngine
from .evaluation import evaluate
from .profile import ProfileStore, UserProfile

__all__ = [
    "FORMATS", "LEVELS", "STRATEGIES", "Recommendation", "RecommenderEngine",
    "ProfileStore", "UserProfile", "evaluate", "load_ratings", "load_resources", "save_ratings",
]
