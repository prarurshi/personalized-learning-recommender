"""Central configuration: file paths, constants and default model weights."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESOURCES_FILE = DATA_DIR / "resources.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
USERS_DIR = DATA_DIR / "users"

LEVELS = ["Beginner", "Intermediate", "Advanced"]
LEVEL_INDEX = {lvl: i for i, lvl in enumerate(LEVELS)}

FORMATS = ["Video Course", "Book", "Tutorial", "Interactive", "Article", "Project"]

# Weights of each signal in the hybrid score (they are re-normalised at run time,
# e.g. collaborative weight is redistributed when a user has not rated anything).
DEFAULT_WEIGHTS = {
    "content": 0.45,        # TF-IDF similarity between profile and resource text
    "collaborative": 0.25,  # item-item collaborative filtering on community ratings
    "knowledge": 0.20,      # rule-based fit: skill level + preferred format
    "popularity": 0.10,     # global resource quality rating
}

# A rating >= LIKE_THRESHOLD counts as "liked", <= DISLIKE_THRESHOLD as "disliked".
LIKE_THRESHOLD = 4
DISLIKE_THRESHOLD = 2
