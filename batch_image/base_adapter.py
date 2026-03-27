"""Abstract base adapter for loading raw datasets into unified format.

To support a new dataset format, subclass BaseAdapter and implement load_entries().
Each entry must have: image_path (str), tags (list[str]), captions (list[str]).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class RawEntry:
    """Unified intermediate format — one per image."""
    image_path: str
    tags: List[str] = field(default_factory=list)
    captions: List[str] = field(default_factory=list)


class BaseAdapter(ABC):
    """Abstract adapter: reads a specific raw format → list[RawEntry]."""

    @abstractmethod
    def load_entries(self) -> List[RawEntry]:
        """Return list of RawEntry from the raw data source."""
        ...
