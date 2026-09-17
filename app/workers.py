"""Background QThread that runs auto-labeling over one or more images so
the GUI thread never blocks on model loading or inference."""
from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import QThread, Signal

from .ocr_engine import OCREngine, OCRNotAvailable


class OcrBatchWorker(QThread):
    progress = Signal(int, int, str)      # current, total, relpath
    image_result = Signal(str, list)      # relpath, list[shape-dict]
    error = Signal(str, str)              # relpath, message
    finished_all = Signal()

    def __init__(self, engine: OCREngine, image_dir: str, relpaths: List[str], parent=None):
        super().__init__(parent)
        self.engine = engine
        self.image_dir = image_dir
        self.relpaths = list(relpaths)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        total = len(self.relpaths)
        for i, relpath in enumerate(self.relpaths):
            if self._cancel:
                break
            self.progress.emit(i + 1, total, relpath)
            full_path = os.path.join(self.image_dir, relpath)
            try:
                shapes = self.engine.detect_and_recognize(full_path)
                self.image_result.emit(relpath, shapes)
            except OCRNotAvailable as e:
                self.error.emit(relpath, str(e))
                break  # engine itself isn't usable; no point retrying per-image
            except Exception as e:
                self.error.emit(relpath, str(e))
                continue  # skip just this image, keep going
        self.finished_all.emit()
