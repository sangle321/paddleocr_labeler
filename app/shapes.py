"""
Data model for a single text-box annotation.

A Shape is always a 4-point quadrilateral (top-left, top-right,
bottom-right, bottom-left, in that order) plus the recognized/edited
text, a "difficult" flag, and an optional OCR confidence score.

This mirrors the record format PaddleOCR / PaddleX use for detection
training data, so a list of Shapes can be dumped straight to JSON and
dropped into a Label.txt file:

    {"transcription": "...", "points": [[x,y],[x,y],[x,y],[x,y]], "difficult": false}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass
class Shape:
    points: List[Point]
    transcription: str = ""
    difficult: bool = False
    score: float = 1.0  # OCR confidence; 1.0 for manually drawn/edited boxes

    def __post_init__(self):
        # Always store plain (float, float) tuples, never numpy scalars /
        # QPointF / lists, so json.dumps and deepcopy behave predictably.
        self.points = [(float(x), float(y)) for x, y in self.points]

    def to_dict(self) -> dict:
        return {
            "transcription": self.transcription,
            "points": [[round(x, 2), round(y, 2)] for x, y in self.points],
            "difficult": bool(self.difficult),
        }

    @staticmethod
    def from_dict(d: dict) -> "Shape":
        pts = [(float(p[0]), float(p[1])) for p in d.get("points", [])]
        return Shape(
            points=pts,
            transcription=str(d.get("transcription", "")),
            difficult=bool(d.get("difficult", False)),
            score=float(d.get("score", 1.0)),
        )
