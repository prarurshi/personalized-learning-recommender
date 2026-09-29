"""Learner profile model and JSON-file based persistence."""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import LEVELS, USERS_DIR


@dataclass
class UserProfile:
    username: str
    name: str = ""
    level: str = "Beginner"
    interests: list = field(default_factory=list)          # topics / keywords
    goals: str = ""                                          # free-text learning goal
    preferred_formats: list = field(default_factory=list)  # empty = any format
    max_duration: Optional[float] = None                     # hours, None = no limit
    free_only: bool = False
    ratings: dict = field(default_factory=dict)             # resource_id -> 1..5
    completed: list = field(default_factory=list)           # resource ids
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        # JSON turns int keys into strings -> normalise back
        self.ratings = {int(k): float(v) for k, v in self.ratings.items()}
        self.completed = [int(x) for x in self.completed]

    def rate(self, resource_id: int, rating: float) -> None:
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        self.ratings[int(resource_id)] = float(rating)

    def mark_completed(self, resource_id: int) -> None:
        if int(resource_id) not in self.completed:
            self.completed.append(int(resource_id))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ratings"] = {str(k): v for k, v in self.ratings.items()}
        return d


class ProfileStore:
    """Stores one JSON file per user in data/users/."""

    def __init__(self, directory: Path = USERS_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def valid_username(username: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_\-]{3,30}", username or ""))

    def _path(self, username: str) -> Path:
        if not self.valid_username(username):
            raise ValueError("username must be 3-30 chars: letters, digits, _ or -")
        return self.directory / f"{username.lower()}.json"

    def exists(self, username: str) -> bool:
        return self._path(username).exists()

    def save(self, profile: UserProfile) -> None:
        self._path(profile.username).write_text(json.dumps(profile.to_dict(), indent=2))

    def load(self, username: str) -> UserProfile:
        path = self._path(username)
        if not path.exists():
            raise FileNotFoundError(f"no profile named '{username}'")
        return UserProfile(**json.loads(path.read_text()))

    def delete(self, username: str) -> None:
        self._path(username).unlink(missing_ok=True)

    def list_users(self) -> list:
        return sorted(p.stem for p in self.directory.glob("*.json"))
