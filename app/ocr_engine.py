"""
Thin wrapper around the `paddleocr` package.

PaddleOCR's Python API changed shape between major versions:

  * 2.x:  ocr = PaddleOCR(use_angle_cls=True, lang='en')
          result = ocr.ocr(img_path, cls=True)
          -> [[ [box, (text, score)], [box, (text, score)], ... ]]

  * 3.x (current, incl. PP-OCRv6): ocr = PaddleOCR(
              lang='en',
              use_doc_orientation_classify=False,
              use_doc_unwarping=False,
              use_textline_orientation=True,
          )
          result = ocr.predict(img_path)
          -> list of dict-like result objects, each exposing
             res['rec_texts'], res['rec_polys'] (or res['dt_polys']),
             res['rec_scores'], res['rec_boxes'] (axis-aligned x1,y1,x2,y2)

This module tries the modern call first and falls back through several
older calling conventions, so the rest of the app never has to know
(or care) which exact paddleocr version is installed. If paddleocr
isn't installed at all, every method raises OCRNotAvailable, and the
GUI degrades to manual-only labeling instead of crashing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_onednn"] = "0"
os.environ["FLAGS_use_mkldnn_bfloat16"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"   # extra safety on newer builds


class OCRNotAvailable(RuntimeError):
    """Raised when the paddleocr package can't be imported or initialized."""


class OCREngine:
    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        use_textline_orientation: bool = True,
        det_model_name: Optional[str] = None,
        rec_model_name: Optional[str] = None,
    ):
        self.lang = lang
        self.device = device
        self.use_textline_orientation = use_textline_orientation
        self.det_model_name = det_model_name
        self.rec_model_name = rec_model_name
        self._ocr: Any = None
        self._init_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # lazy initialization
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except OCRNotAvailable:
            return False

    def _load(self) -> None:
        if self._ocr is not None:
            return
        if self._init_error is not None:
            raise self._init_error

        import os
        # Force-disable OneDNN / MKLDNN – required to avoid the
        # "ConvertPirAttribute2RuntimeAttribute not support" crash
        # that appears with many recent paddlepaddle + paddleocr combinations.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_onednn"] = "0"
        os.environ["FLAGS_use_mkldnn_bfloat16"] = "0"

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            self._init_error = OCRNotAvailable(
                "The 'paddleocr' package isn't installed, so automatic "
                "labeling is unavailable. Install it with:\n"
                "    pip install paddlepaddle paddleocr\n"
                "You can still draw and edit boxes manually."
            )
            raise self._init_error from e

        model_kwargs = {}
        if self.det_model_name:
            model_kwargs["text_detection_model_name"] = self.det_model_name
        if self.rec_model_name:
            model_kwargs["text_recognition_model_name"] = self.rec_model_name

        # Progressively simpler kwarg sets, newest/most-specific first.
        # Whichever paddleocr version is installed, one of these should work.
        candidate_kwargs = [
            dict(
                lang=self.lang, device=self.device,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=self.use_textline_orientation,
                enable_mkldnn=False,
                **model_kwargs,
            ),
            dict(
                lang=self.lang,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=self.use_textline_orientation,
                enable_mkldnn=False,
                **model_kwargs,
            ),
            dict(lang=self.lang, use_angle_cls=self.use_textline_orientation, enable_mkldnn=False, ),
            dict(lang=self.lang),
            dict(),
        ]

        last_err: Optional[Exception] = None
        for kwargs in candidate_kwargs:
            try:
                self._ocr = PaddleOCR(**kwargs)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                # Non-TypeError failures (bad device string, missing model,
                # network error downloading weights, etc.) won't be fixed by
                # trying a different kwarg combination, so stop here.
                self._init_error = OCRNotAvailable(f"Could not initialize PaddleOCR: {e}")
                raise self._init_error from e

        self._init_error = OCRNotAvailable(f"Could not initialize PaddleOCR: {last_err}")
        raise self._init_error

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def detect_and_recognize(self, image) -> List[Dict[str, Any]]:
        """`image` is a file path (str) or a BGR numpy array.
        Returns a list of plain dicts: {"points": [[x,y]x4], "transcription": str, "score": float}
        so results can safely cross a QThread signal boundary."""
        self._load()

        if hasattr(self._ocr, "predict"):
            try:
                results = self._ocr.predict(image)
                return self._parse_modern_result(results)
            except Exception:
                pass  # fall back to the legacy call below

        result = self._ocr.ocr(image)
        return self._parse_legacy_result(result)

    def recognize_only(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Re-run the full pipeline on an already-cropped text region and
        return (text, score) for the first thing it finds. Used to
        re-recognize a single box's text after the user reshapes it."""
        shapes = self.detect_and_recognize(crop_bgr)
        if not shapes:
            return "", 0.0
        if len(shapes) == 1:
            return shapes[0]["transcription"], shapes[0]["score"]
        # Occasionally a crop is detected as more than one line; join them
        # left-to-right rather than silently dropping text.
        shapes.sort(key=lambda s: min(p[0] for p in s["points"]))
        text = " ".join(s["transcription"] for s in shapes if s["transcription"])
        score = min(s["score"] for s in shapes)
        return text, score

    # ------------------------------------------------------------------
    # result parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap(res: Any) -> Dict[str, Any]:
        """Result objects are dict-like in every 3.x version seen so far,
        but some tooling nests the real payload one level under a 'res'
        key when printing. Handle both shapes."""
        if hasattr(res, "json"):
            try:
                data = res.json
                if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
                    return data["res"]
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        if isinstance(res, dict):
            if "rec_texts" not in res and isinstance(res.get("res"), dict):
                return res["res"]
            return res
        # Fall back to attribute access for non-dict result objects.
        return {k: getattr(res, k) for k in
                ("rec_texts", "rec_polys", "dt_polys", "rec_scores", "rec_boxes")
                if hasattr(res, k)}

    @classmethod
    def _parse_modern_result(cls, results) -> List[Dict[str, Any]]:
        shapes: List[Dict[str, Any]] = []
        for res in results:
            data = cls._unwrap(res)
            texts = list(data.get("rec_texts") or [])
            scores = list(data.get("rec_scores") or [])
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys")
            if polys is None:
                # last resort: expand axis-aligned [x1,y1,x2,y2] boxes into 4 points
                boxes = data.get("rec_boxes")
                polys = []
                if boxes is not None:
                    for b in boxes:
                        x1, y1, x2, y2 = [float(v) for v in b]
                        polys.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            for i, poly in enumerate(polys or []):
                pts = [[float(x), float(y)] for x, y in poly]
                text = texts[i] if i < len(texts) else ""
                score = float(scores[i]) if i < len(scores) else 1.0
                shapes.append({"points": pts, "transcription": text, "score": score})
        return shapes

    @staticmethod
    def _parse_legacy_result(result) -> List[Dict[str, Any]]:
        shapes: List[Dict[str, Any]] = []
        if not result:
            return shapes
        # The 2.x .ocr() call wraps one image's lines in an extra outer list.
        lines = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
        for line in lines or []:
            if not line:
                continue
            box, rec = line[0], line[1]
            text, score = (rec[0], rec[1]) if isinstance(rec, (list, tuple)) else (rec, 1.0)
            pts = [[float(x), float(y)] for x, y in box]
            shapes.append({"points": pts, "transcription": text, "score": float(score)})
        return shapes
