# Fine-Tuning PP-OCRv6 Small (Text Detection) on Kaggle — Step-by-Step

> **A quick heads-up on versions:** PP-OCRv6 is a real, current release (PaddleOCR v3.7.0, shipped June 11, 2026) — three tiers: **tiny** (edge/IoT), **small** (mobile/desktop, ~9.6 MB, 84.1% Hmean on PaddleOCR's internal benchmark), and **medium** (server, highest accuracy). This guide fine-tunes **PP-OCRv6_small_det**, the detector-only model, using PaddleOCR's own training scripts — the same workflow the official docs use, just pointed at your dataset.
>
> One thing I couldn't 100%-verify from outside a live environment: the exact config filename inside `configs/det/PP-OCRv6/`. Every prior version follows the pattern `configs/det/{VERSION}/{MODEL_NAME}.yml`, so it's very likely `configs/det/PP-OCRv6/PP-OCRv6_small_det.yml` — but **Step 4.3 has you list that folder and confirm the real name before you run anything**, so this can't silently break your training command.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Confirm your label format](#2-confirm-your-label-format)
3. [Package and upload your dataset to Kaggle](#3-package-and-upload-your-dataset-to-kaggle)
4. [Set up the Kaggle notebook](#4-set-up-the-kaggle-notebook)
5. [Point the config at your data](#5-point-the-config-at-your-data)
6. [Optional: 2-minute pipeline sanity check](#6-optional-2-minute-pipeline-sanity-check)
7. [Fine-tune](#7-fine-tune)
8. [Evaluate](#8-evaluate)
9. [Export to an inference model](#9-export-to-an-inference-model)
10. [Quick visual test](#10-quick-visual-test)
11. [Save/download your model](#11-savedownload-your-model)
12. [Fitting this into Kaggle's session limits](#12-fitting-this-into-kaggles-session-limits)
13. [Troubleshooting](#13-troubleshooting)
14. [Next steps](#14-next-steps)
15. [Sources](#15-sources)

---

## 1. Before you start

**What you need:**
- A Kaggle account with phone verification (required for GPU access)
- Your `paddle_dataset` folder (as in your screenshots): `images/train/`, presumably `images/val/`, `train.txt`, `val.txt`

**Optional but useful — see the stock model's baseline first.** Before investing in fine-tuning, it's worth seeing how the *un-tuned* PP-OCRv6_small_det already does on your documents, so you can judge how much fine-tuning actually buys you:

```python
# In any Python environment with `pip install paddleocr` (3.7+)
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv6_small_det")
output = model.predict("path/to/one_of_your_scans.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./baseline_output/")
```

If it already does well, you may only need a light fine-tune (fewer epochs); if it misses a lot of your form fields/tables, a fuller fine-tune is worth it.

---

## 2. Confirm your label format

PaddleOCR's detection trainer expects `train.txt` / `val.txt` where **each line** is:

```
<relative_image_path>\t<json_list_of_boxes>
```

Example (tab between the two parts):

```
images/train/0b1d7817-1135-4-7cc-b7d1-a9aa25d5bf38.jpg	[{"transcription": "Invoice No.", "points": [[120, 40], [230, 40], [230, 65], [120, 65]]}, {"transcription": "###", "points": [[10, 10], [50, 10], [50, 30], [10, 30]]}]
```

Rules that matter:
- `points` = 4 corners `[x, y]`, clockwise starting from top-left.
- `transcription` = the text in that box. If a box should be **ignored** during detection training (illegible, or you don't have a transcription), set `transcription` to `"###"` — PaddleOCR skips those but still uses them to avoid punishing the model for detecting real text there.
- Image paths in the txt file are relative to whatever you set as `data_dir` later (so `images/train/...` implies `data_dir` should point at the folder that *contains* `images/`).

Since your folder is already named `paddle_dataset/images/train` + `train.txt`/`val.txt` at the same level, it looks like you've already built this in the expected shape. You'll do a live parse-check in Step 5.2 once the files are on Kaggle, to be sure before a multi-hour training run depends on it.

---

## 3. Package and upload your dataset to Kaggle

1. On your machine, zip the **contents** of `paddle_dataset` (the `images` folder + `train.txt` + `val.txt`) — not the parent folder that contains `paddle_dataset`. This keeps the path shallow once Kaggle extracts it.
2. Go to **kaggle.com → Datasets → New Dataset**, upload the zip, give it a name (e.g. `paddle-det-dataset`), and create it. Kaggle unzips it automatically.
3. Note the dataset's URL slug (e.g. `your-username/paddle-det-dataset`) — you'll attach it to your notebook next.

> Whatever the exact internal nesting ends up being, **Step 4.2 has you `ls` the real path** before you rely on it — no need to guess correctly here.

---

## 4. Set up the Kaggle notebook

### 4.1 Create the notebook
- **New Notebook**.
- Right sidebar → **Accelerator** → `GPU T4 x2` (or `GPU P100` if that's what's available to you).
- Right sidebar → **Internet** → **On** (required — you'll `pip install`, `git clone`, and download pretrained weights from `bcebos.com`).
- Right sidebar → **Add Input** → search for your dataset from Step 3 → add it.

### 4.2 Cell 1 — check your GPU and confirm the real input path

```python
!nvidia-smi
!ls /kaggle/input/
```

Note two things from the output:
- The **CUDA Version** shown top-right of `nvidia-smi` (this is the *driver's* max supported version, used in the next cell).
- The exact folder name under `/kaggle/input/` — this is your real `DATA_DIR` for every step below. Then run:

```python
!ls /kaggle/input/<your-dataset-folder>/
```
to confirm you see `images/`, `train.txt`, `val.txt` directly (adjust `DATA_DIR` in Step 5 if there's an extra nested folder).

### 4.3 Cell 2 — install PaddlePaddle (GPU) and the PaddleOCR repo

```bash
# Pick the tag matching (or just below) the CUDA Version nvidia-smi reported.
# cu126 is a safe default for current Kaggle GPU images; swap to cu118 / cu129 / cu130 if needed.
!python -m pip install -q paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

!git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR
%cd /kaggle/working/PaddleOCR
!pip install -q -r requirements.txt
```

Verify the install:

```python
import paddle
paddle.utils.run_check()
```

You want to see it confirm PaddlePaddle is installed and detects your GPU.

### 4.4 Cell 3 — confirm the PP-OCRv6 small-det config filename

```bash
!ls configs/det/PP-OCRv6/
```

You're looking for something like `PP-OCRv6_small_det.yml`. If the folder or filename differs slightly, find it with:

```bash
!find configs -iname "*ocrv6*small*det*"
```

**Use whatever path this reveals in every `-c` flag below** — I'll write `configs/det/PP-OCRv6/PP-OCRv6_small_det.yml` throughout, but treat that as "the config path you just confirmed," not gospel.

### 4.5 Cell 4 — download the PP-OCRv6_small_det pretrained weights

This is the actual trained detector (not just a backbone init) you'll continue training from:

```bash
!wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams \
  -O /kaggle/working/PP-OCRv6_small_det_pretrained.pdparams
```

---

## 5. Point the config at your data

### 5.1 Set your paths once, reuse everywhere

```python
DATA_DIR = "/kaggle/input/<your-dataset-folder>/"   # from Step 4.2
TRAIN_TXT = DATA_DIR + "train.txt"
VAL_TXT = DATA_DIR + "val.txt"
print(DATA_DIR, TRAIN_TXT, VAL_TXT, sep="\n")
```

### 5.2 Sanity-check the label format before committing to a long run

```python
import json

with open(TRAIN_TXT, encoding="utf-8") as f:
    line = f.readline()

img_path, label_json = line.rstrip("\n").split("\t", 1)
boxes = json.loads(label_json)

print("image:", img_path)
print("num boxes:", len(boxes))
print("first box:", boxes[0])
```

You want output like:
```
image: images/train/0b1d7817-....jpg
num boxes: 14
first box: {'transcription': '...', 'points': [[..], [..], [..], [..]]}
```

If this throws a `ValueError` (no tab) or `JSONDecodeError`, your label file isn't in this format yet and needs converting before training — PaddleOCR's `ppocr/utils/gen_label.py` (in the repo you just cloned) or the PPOCRLabel annotation tool are the standard ways to produce it.

Also confirm the validation images actually exist:
```bash
!ls {DATA_DIR}images/ | head
```
(only `train` was visible in your screenshot — make sure `val` is there too, since `val.txt` references it.)

---

## 6. Optional: 2-minute pipeline sanity check

Before trusting a multi-hour run to your real data, it's worth proving the training pipeline itself works end-to-end using PaddleOCR's tiny official demo set — isolates "my environment is broken" from "my dataset/config is the problem":

```bash
!wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar
!tar -xf ocr_det_dataset_examples.tar

!python3 tools/train.py -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.pretrained_model=/kaggle/working/PP-OCRv6_small_det_pretrained.pdparams \
    Global.epoch_num=1 \
    Global.save_model_dir=/kaggle/working/output/sanity_check \
    Train.dataset.data_dir=./ocr_det_dataset_examples \
    Train.dataset.label_file_list='[./ocr_det_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./ocr_det_dataset_examples \
    Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]'
```

If this runs a full epoch and prints an eval Hmean without erroring, your environment and config path are good — move on to your real data.

---

## 7. Fine-tune

```bash
!python3 tools/train.py -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.pretrained_model=/kaggle/working/PP-OCRv6_small_det_pretrained.pdparams \
    Global.save_model_dir=/kaggle/working/output/PP-OCRv6_small_det \
    Global.epoch_num=50 \
    Optimizer.lr.learning_rate=0.0001 \
    Train.loader.batch_size_per_card=8 \
    Train.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Train.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/train.txt]' \
    Eval.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Eval.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/val.txt]'
```

**Why these values:**
| Override | Why |
|---|---|
| `Global.epoch_num=50` | A generous starting budget. PaddleOCR always keeps `best_accuracy.pdparams` based on eval score regardless of total epochs, so it's safe to set this high and just stop the cell early (or let it finish) once the printed eval Hmean plateaus. |
| `Optimizer.lr.learning_rate=0.0001` | Fine-tuning generally wants a much lower LR than training from scratch (PaddleOCR's from-scratch configs are typically ~1e-3); dropping an order of magnitude avoids destroying the pretrained weights in the first few steps. |
| `Train.loader.batch_size_per_card=8` | A safe starting point for a 16 GB T4 with this small a model. Raise it if GPU memory usage looks low (check with `!nvidia-smi` mid-run in a second cell), lower it if you hit an out-of-memory error. |

**If you have `GPU T4 x2` enabled**, you can use both GPUs to roughly halve wall-clock time:
```bash
!python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py \
    -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.pretrained_model=/kaggle/working/PP-OCRv6_small_det_pretrained.pdparams \
    Global.save_model_dir=/kaggle/working/output/PP-OCRv6_small_det \
    Global.epoch_num=50 \
    Optimizer.lr.learning_rate=0.0001 \
    Train.loader.batch_size_per_card=8 \
    Train.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Train.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/train.txt]' \
    Eval.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Eval.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/val.txt]'
```

Watch the logs for the periodic eval lines (Hmean/precision/recall on your val set) — that's your real signal, not the training loss alone.

---

## 8. Evaluate

Once training finishes (or you stop it), evaluate the best checkpoint explicitly:

```bash
!python3 tools/eval.py -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.pretrained_model=/kaggle/working/output/PP-OCRv6_small_det/best_accuracy.pdparams \
    Eval.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Eval.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/val.txt]'
```

This prints precision / recall / Hmean for `best_accuracy.pdparams` on your val set.

---

## 9. Export to an inference model

Training checkpoints (`.pdparams`) aren't directly usable by the normal inference API — export a static-graph inference model first:

```bash
!python3 tools/export_model.py -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.pretrained_model=/kaggle/working/output/PP-OCRv6_small_det/best_accuracy.pdparams \
    Global.save_inference_dir=/kaggle/working/PP-OCRv6_small_det_infer/
```

This produces:
```
/kaggle/working/PP-OCRv6_small_det_infer/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

---

## 10. Quick visual test

This uses the separate **`paddleocr` pip package** (the packaged inference API — different from the git repo you cloned for training, which only has the training *scripts*):

```bash
!pip install -q "paddleocr>=3.7"
```

```python
from paddleocr import TextDetection

model = TextDetection(model_dir="/kaggle/working/PP-OCRv6_small_det_infer/")
output = model.predict(DATA_DIR + "images/val/SOME_VAL_IMAGE.jpg", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="/kaggle/working/test_output/")
```

Then display the saved visualization inline:

```python
from PIL import Image
import glob
img_path = glob.glob("/kaggle/working/test_output/*")[0]
display(Image.open(img_path))
```

Compare this against the "before fine-tuning" baseline from Step 1 to see the actual improvement on your document types.

---

## 11. Save/download your model

Anything in `/kaggle/working` only persists past your session if you **Save Version → Save & Run All (Commit)** — that becomes a Notebook Output you can download as a zip afterward, or reuse directly as the input to another notebook.

To make the exported model easy to grab or reuse:
```python
import shutil
shutil.make_archive("/kaggle/working/PP-OCRv6_small_det_finetuned", "zip", "/kaggle/working/PP-OCRv6_small_det_infer")
```
Then either download it from the notebook's Output tab after committing, or click **New Dataset** from your notebook's output to publish the fine-tuned model as its own reusable Kaggle Dataset.

---

## 12. Fitting this into Kaggle's session limits

Kaggle GPU sessions are time-boxed (commonly reported around 9 hours per session, with a rolling weekly GPU quota — check **Settings → your account** for your exact current numbers, since these have shifted before). Two practical implications:

- **Use "Save & Run All (Commit)"** for the actual training cell rather than running it interactively — a committed run keeps going in the background even if you close the tab, and its output is what actually gets saved.
- **If a single run might not finish**, checkpoints let you resume: PaddleOCR saves a `latest.pdparams`/`latest.pdopt` pair in `Global.save_model_dir`. In a new session (with your previous output attached as an input), resume like this — note `Global.checkpoints` replaces `Global.pretrained_model` here, since it restores the optimizer state and epoch count too, not just weights:

```bash
!python3 tools/train.py -c configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
    -o Global.checkpoints=/kaggle/input/<your-previous-output>/output/PP-OCRv6_small_det/latest \
    Global.save_model_dir=/kaggle/working/output/PP-OCRv6_small_det \
    Global.epoch_num=50 \
    Optimizer.lr.learning_rate=0.0001 \
    Train.loader.batch_size_per_card=8 \
    Train.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Train.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/train.txt]' \
    Eval.dataset.data_dir=/kaggle/input/<your-dataset-folder>/ \
    Eval.dataset.label_file_list='[/kaggle/input/<your-dataset-folder>/val.txt]'
```

Given how small this model is (~2.5M parameters for the detector alone), a full fine-tune on a modest custom dataset will often finish comfortably inside one session — treat this section as a fallback, not the default plan.

---

## 13. Troubleshooting

- **Out of memory** → lower `Train.loader.batch_size_per_card` (try 4, then 2).
- **Loss goes to `NaN`** → lower `Optimizer.lr.learning_rate` further (try `0.00005`), and double check Step 5.2's sanity check — malformed boxes (zero-area, out-of-image coordinates) are a common cause.
- **Small text / dense table cells getting missed** → the config's resize/limit settings control the max input resolution during detection; check `!cat configs/det/PP-OCRv6/PP-OCRv6_small_det.yml` for the resize transform and consider raising the side-length limit at the cost of speed/memory.
- **An `-o` override throws a key error** → run `!cat configs/det/PP-OCRv6/PP-OCRv6_small_det.yml` and confirm the exact nested key path in *this* file — occasionally a key gets renamed between releases.
- **`pip install -r requirements.txt` fails on one package** → Kaggle's base image already has most scientific-Python deps (numpy, opencv, scipy); a single failing line rarely blocks the rest — rerun without it or `pip install` it separately with a compatible version.

---

## 14. Next steps

This guide only fine-tunes the **detector**. For a fully custom OCR pipeline on your documents, the recognizer (`PP-OCRv6_small_rec`) usually benefits from its own fine-tune too, especially if your text uses unusual fonts, tables, or a language mix — same overall workflow, but via the [Text Recognition Tutorial](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html) instead of the detection one.

---

## 15. Sources

- [PP-OCRv6 Introduction — PaddleOCR Docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- [Text Detection Module Usage Guide — PaddleOCR Docs](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html) (models table, pretrained weight links, train/eval/export commands)
- [PaddleOCR v3.7.0 Release Notes — GitHub](https://github.com/PaddlePaddle/PaddleOCR/releases)
- [PaddleOCR Installation Guide (3.x)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/installation.en.md)
- [PaddleOCR Quick Start — paddlepaddle-gpu install pattern](https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html)
- [PaddlePaddle Installation Guide — CUDA wheel tags](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
- [PaddleOCR detection label format spec](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/detection_en.md)
- [Kaggle: Efficient GPU Usage docs](https://www.kaggle.com/docs/efficient-gpu-usage)
