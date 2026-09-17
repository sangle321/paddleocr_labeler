# PaddleOCR Label Studio

A small desktop GUI for building fine-tuning data for PaddleOCR (PP-OCRv6 and
earlier). Point it at a folder of images, let PaddleOCR auto-label the text
boxes, then correct the boxes and text by hand. It saves everything in the
exact format PaddleOCR/PaddleX detection and recognition training expect, so
the output folder is ready to train on.

Workflow: **Open folder → Auto-Label → drag boxes / fix text / mark
unreadable text as "difficult" → Save → Export dataset.**

## What it looks like

- **Left:** the list of images in the folder (a checkmark means it has saved boxes).
- **Middle:** the current image, with editable quadrilateral boxes drawn on top.
- **Right:** the list of boxes for the current image — edit the recognized
  text inline, flag a box as "difficult", or delete it.

## 1. Install

```bash
# 1) The deep-learning framework PaddleOCR runs on
pip install paddlepaddle
# ...or, if you have an NVIDIA GPU set up for CUDA:
# pip install paddlepaddle-gpu

# 2) PaddleOCR itself, plus the GUI's other dependencies
pip install paddleocr PySide6 opencv-python numpy

# (equivalently: pip install -r requirements.txt)
```

Requires Python 3.9–3.12. PaddleOCR downloads its model weights (~100–300 MB)
the first time you run auto-labeling, so make sure you have internet access
for that first run; after that they're cached locally.

**You don't strictly need PaddleOCR installed to use this tool.** If it's
missing, the app still opens and works — you just draw every box by hand
instead of starting from an auto-labeled one.

## 2. Run it

```bash
python main.py
# or open a folder immediately:
python main.py /path/to/your/images
```

## 3. Label your images

1. **File → Open Image Folder…** and pick a folder of `.jpg` / `.png` / `.bmp`
   / `.tif` / `.webp` images.
2. Pick the **OCR language** in the right-hand panel (English, Chinese,
   Vietnamese, etc. — see the dropdown for the full list).
3. **OCR → Auto-Label Current Image** (or **Auto-Label All Images…** to run
   the whole folder in one go — it runs in the background so the app stays
   responsive, and shows a progress bar you can cancel). "Auto-Label All"
   asks whether to skip images that already have saved boxes.
4. Fix what PaddleOCR got wrong:
   - **Drag a corner handle** to reshape a box.
   - **Drag inside a box** to move the whole thing.
   - **Draw box** (toolbar/`Ctrl+B`): click-drag to add a new axis-aligned box.
   - **Draw quad** (`Ctrl+Shift+B`): click 4 corners in order, for rotated or
     skewed text a plain rectangle won't fit.
   - Click a box (or its row in the table) to select it; **Delete/Backspace**
     removes the selected box.
   - Edit the recognized text directly in the table's **Text** column.
   - Check **Difficult** for text that's unreadable or shouldn't be used for
     recognition training (it's still kept for detection training).
   - **OCR → Re-recognize Selected Box** re-runs recognition on just that
     crop — handy after you've reshaped a box PaddleOCR got wrong.
   - **OCR → Auto Re-recognize After Editing Box** (on by default) does that
     re-recognition automatically the moment you finish dragging a corner or
     moving a box, so the text never goes stale relative to the box. Undoing
     the drag (`Ctrl+Z`) restores the old text along with the old geometry as
     one step. Turn this off in the OCR menu if you'd rather nudge boxes
     without a fresh OCR guess overwriting text you've hand-corrected.
5. **Ctrl+S** saves the current image's boxes to `Label.txt` in the image
   folder. The app also auto-saves the current image whenever you switch to
   another one or close the window, so you generally won't lose work.

Mouse wheel zooms, middle-click-drag pans, `Ctrl+0` fits the image to the
window, `Ctrl+Z` / `Ctrl+Y` undo/redo box position, creation and deletion
(text edits aren't tracked in undo history). **Help → Keyboard Shortcuts**
has the full list.

## 4. What gets saved

Everything lives in the image folder itself:

**`Label.txt`** — one line per labeled image:

```
receipt_014.jpg\t[{"transcription": "TOTAL $42.50", "points": [[88,201],[310,201],[310,238],[88,238]], "difficult": false}, ...]
```

This is the standard PaddleOCR/PaddleX detection-training format: a relative
image path, a tab, then a JSON array of boxes (4 corner points each, plus the
transcribed text and whether it's marked difficult). It's already in the
right shape to be used as-is for detection fine-tuning.

## 5. Exporting for recognition training and beyond

Recognition training needs individually cropped text images rather than
full pages, so use **File** menu:

- **Export Recognition Dataset…** — perspective-crops every non-difficult,
  non-empty box out of its source image (handling rotated boxes correctly)
  into a folder containing `crop_img/*.jpg` plus `rec_gt_train.txt` /
  `rec_gt_val.txt` (a 90/10 split by default), each line `crop path<TAB>text`.
- **Export Detection Train/Val Split…** — splits `Label.txt` into
  `Label_train.txt` / `Label_val.txt`.
- **Build Character Dictionary…** — collects every character that appears
  anywhere in your transcriptions into a one-char-per-line dictionary file.
  Useful if you're labeling a language/script (accents, Vietnamese
  diacritics, currency symbols, etc.) that isn't fully covered by
  PaddleOCR's default recognition dictionary — pass this file as
  `character_dict_path` when you fine-tune so training doesn't silently
  drop characters your data actually uses.

## 6. Using this data to fine-tune PaddleOCR

The GUI only *labels* data — actually training a model uses PaddleOCR's own
training code, which is a separate install from the `paddleocr` pip package:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -e .
```

Current PaddleOCR (v3.x, including PP-OCRv6) trains models through **PaddleX**,
its unified low-code training tool, with one command per model:

```bash
# Detection, e.g. fine-tuning a PP-OCRv6 detection model
python main.py -c paddlex/configs/text_detection/PP-OCRv6_mobile_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/path/to/your/image_folder

# Recognition
python main.py -c paddlex/configs/text_recognition/PP-OCRv6_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/path/to/your/exported_recognition_folder \
    -o Train.dataset.character_dict_path=/path/to/custom_dict.txt
```

A few notes:
- Check `paddlex/configs/text_detection/` and `.../text_recognition/` in your
  clone for the exact config filename you want (naming/available tiers can
  shift between releases) — pick the tier (tiny/small/medium, or
  mobile/server on older versions) closest to your deployment target.
- PaddleOCR's own guidance: real fine-tuning data helps most in
  small-but-meaningful amounts — roughly 500+ images for detection and
  5,000+ text-line crops for recognition — mixed with some general-scene
  data so the model doesn't overfit to just your scenario. Small datasets
  especially benefit from a low learning rate and fewer epochs than
  training from scratch.
- The classic `tools/train.py -c configs/det/...yml` / `tools/eval.py`
  workflow from older PaddleOCR versions still exists in the repo and reads
  the same `Label.txt` format, if you're targeting an older PP-OCR version
  instead of PaddleX.
- Official docs: <https://paddlepaddle.github.io/PaddleOCR/> and the
  PaddleX pipeline guide at
  <https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/pipeline_develop_guide.html>
  have the fully up-to-date flags and config layout.

## Project layout

```
paddleocr_labeler/
├── main.py              entry point
├── requirements.txt
└── app/
    ├── shapes.py         the Shape (4-point box) data class
    ├── label_store.py    Label.txt read/write
    ├── ocr_engine.py     wraps paddleocr; works across its 2.x/3.x APIs
    ├── workers.py        background QThread that runs OCR without freezing the UI
    ├── canvas_view.py     the interactive image/box canvas
    ├── export.py         recognition-crop export, det train/val split, char dict
    └── main_window.py     wires everything together
```

## Troubleshooting

- **"OCR error" popup mentioning `paddleocr` isn't installed** — see step 1;
  the app still works for manual-only labeling either way.
- **PaddleOCR initializes but gives an argument/TypeError in the log** —
  `ocr_engine.py` already tries several constructor-argument combinations to
  cover different paddleocr versions, but if your installed version is
  unusual, open `OCREngine._load()` and adjust `candidate_kwargs` to match
  the constructor your version expects (check `help(PaddleOCR)` in a Python
  shell). Everything downstream reads a plain list of `{points, transcription,
  score}` dicts, so this is the only place version quirks should ever need
  touching.
- **First run is slow / needs internet** — that's PaddleOCR downloading model
  weights to `~/.paddleocr/`; later runs use the cache.
- **Nothing detected on a very small/blurry crop when re-recognizing a box** —
  try widening the box slightly before re-running; recognition needs some
  margin around the glyphs.
