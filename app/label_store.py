"""
Reads and writes a PaddleOCR / PaddleX-style detection label file:

    <relative_image_path>\t[{"transcription": "...", "points": [[x,y]x4], "difficult": false}, ...]

one line per image, tab-separated, JSON array of box records after the tab.
Putting this file (by default named Label.txt) next to the images and
pointing PaddleOCR/PaddleX's detection training config at the image folder
is enough to use it for fine-tuning -- see the README for details.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .shapes import Shape


class LabelStore:
    def __init__(self, image_dir: str, label_filename: str = "Label.txt"):
        self.image_dir = image_dir
        self.label_path = os.path.join(image_dir, label_filename)
        self.data: Dict[str, List[Shape]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.label_path):
            return
        with open(self.label_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                if "\t" not in line:
                    continue
                relpath, json_str = line.split("\t", 1)
                try:
                    items = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                self.data[relpath] = [Shape.from_dict(it) for it in items]

    def get(self, relpath: str) -> List[Shape]:
        return self.data.get(relpath, [])

    def set(self, relpath: str, shapes: List[Shape]) -> None:
        self.data[relpath] = list(shapes)

    def all_relpaths(self) -> List[str]:
        return list(self.data.keys())

    def save(self) -> None:
        """Write the whole label file atomically (write to a temp file,
        then rename over the original) so a crash mid-write can never
        leave a half-written Label.txt behind."""
        tmp_path = self.label_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for relpath, shapes in self.data.items():
                items = [s.to_dict() for s in shapes]
                f.write(f"{relpath}\t{json.dumps(items, ensure_ascii=False)}\n")
        os.replace(tmp_path, self.label_path)
