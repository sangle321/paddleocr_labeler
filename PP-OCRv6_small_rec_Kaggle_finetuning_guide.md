# Fine-tuning PP-OCRv6_small_rec on Kaggle — Full Guide

**Your setup:** ~400,000 recognition samples in the standard PaddleOCR format (`paddle_ocr_data/train_label.txt`, `val_label.txt`, `images/`), Japanese text, target model `PP-OCRv6_small_rec`, training on Kaggle.

## Important framing before you start

PP-OCRv6 shipped in June 2026 (PaddleOCR v3.7.0). The **small** and **medium** tiers already support Japanese natively as part of a single 50-language model (the **tiny** tier does *not* include Japanese — good thing you picked small). This matters a lot for how your fine-tune will behave:

- You are doing **domain adaptation**, not teaching the model a new script from scratch. Both the backbone *and* the character head are already meaningful for Japanese.
- Compare this to someone fine-tuning PP-OCRv6_small_rec for a language it *doesn't* ship with (e.g. Arabic): their character head is randomly re-initialized and accuracy is near-zero for the first 1–2 epochs. You shouldn't see that — your model should show reasonable accuracy from very early on, and likely needs **fewer epochs** than a cold-start run.
- The main risks for you are different: catastrophic forgetting from too-high a learning rate, overfitting to a narrow document domain (addresses/company names look repetitive across 400k samples), and out-of-vocabulary kanji that aren't in the default dictionary.

Quick model facts (small tier): ~7.7M params, ~20.4MB on disk, CTC+NRTR multi-head decoder over a PPLCNetV4 backbone with a "LightSVTR" recognition neck.

---

## Phase 0 — Sanity-check your data before uploading anything

Your screenshot already shows the right shape (`images/xxxx.jpg` **[tab]** `label text`, one per line) — that's exactly the PaddleOCR `SimpleDataSet` label format. Before you burn any Kaggle GPU hours, run this locally or in a throwaway Kaggle CPU session:

```python
import os

def load_labels(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:   # utf-8-sig also silently handles a BOM if Notepad added one
        for i, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if "\t" not in line:
                print(f"Line {i} has no tab separator: {line[:60]!r}")
                continue
            img_path, text = line.split("\t", 1)
            rows.append((img_path, text))
    return rows

train = load_labels("paddle_ocr_data/train_label.txt")
val   = load_labels("paddle_ocr_data/val_label.txt")
print(f"train: {len(train)} lines | val: {len(val)} lines")

# 1) Leakage check — same crop shouldn't appear in both splits
train_imgs, val_imgs = {p for p, _ in train}, {p for p, _ in val}
overlap = train_imgs & val_imgs
print(f"train/val image overlap: {len(overlap)}")

# 2) Missing files check
missing = [p for p, _ in (train[:5000] + val)  # sample train for speed, check all of val
           if not os.path.exists(os.path.join("paddle_ocr_data", p))]
print(f"missing image files (sampled): {len(missing)}")

# 3) Label length — decides your max_text_length setting
lengths = [len(t) for _, t in train]
print(f"label length: max={max(lengths)}, p99={sorted(lengths)[int(len(lengths)*0.99)]}")

# 4) Empty / suspicious labels
empties = sum(1 for _, t in train if len(t.strip()) == 0)
print(f"empty labels: {empties}")
```

If `train/val image overlap` is non-zero, your validation accuracy will be optimistic — worth deduplicating. If `p99` label length is above ~25, you'll need to raise `max_text_length` in the config (see Phase 3).

**Character coverage check** (this is the one that actually matters for accuracy). Real-world Japanese OCR data — especially addresses and company names, which is exactly what your sample shows — routinely contains kanji outside a generic dictionary:

```python
# Point this at the dict file referenced by Global.character_dict_path
# in the PP-OCRv6_small_rec.yml you'll download in Phase 2 (see note below).
dict_path = "PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt"  # confirm the real filename in your copy of the yml

with open(dict_path, encoding="utf-8") as f:
    dict_chars = set(line.rstrip("\n") for line in f)

data_chars = set()
for _, text in train + val:
    data_chars.update(text)

missing_chars = sorted(data_chars - dict_chars)
print(f"{len(missing_chars)} characters in your labels are NOT in the dictionary")
print("".join(missing_chars[:300]))
```

- If `missing_chars` is empty or tiny (a handful of odd symbols), you're fine — fine-tune with the stock dictionary and skip straight to Phase 1.
- If it's a real list of kanji, check how many *label lines* actually contain one before deciding — it's often a small fraction, in which case just filtering those lines out of `train_label.txt` is the simplest fix.
- If it's common enough to matter, you can extend the dictionary — **but not by editing the `.txt` file alone.** PaddleOCR's loader matches pretrained weights to the new model by tensor shape. The CTC head's output layer and the NRTR head's output projection + embedding table are all sized by dictionary length — grow the dictionary and those tensors no longer match what's saved in `PP-OCRv6_small_rec_pretrained.pdparams`. The loader can't partially fill a bigger tensor from a smaller one, so it just drops the whole thing and reinitializes that layer from random — silently discarding pretrained knowledge for every one of the ~18k+ characters it already knew, not just the handful you're adding.

  The proper fix is to resize the checkpoint itself before training starts: build a new `.pdparams` where the head tensors are already the bigger size, with the old weights copied into their old positions and only the new rows randomly initialized. Here's a function that does this — it identifies the tensors to grow generically (by shape *and* a `"head"` name filter, rather than guessing exact parameter names I can't fully verify without your actual checkpoint file), and reports exactly what it touched so you can sanity-check it:

  ```python
  import numpy as np
  import paddle

  def resize_vocab_checkpoint(old_ckpt_path, new_ckpt_path, old_dict_len, new_dict_len,
                               name_filter="head", offsets=(0, 1, 2, 3), init_std=0.02, seed=0):
      """
      Grow every classifier/embedding tensor in a PaddleOCR .pdparams checkpoint
      whose size depends on dictionary length, copying old weights into the
      matching old positions instead of discarding them.

      A tensor is resized only if BOTH hold:
        1) its key contains `name_filter` (default "head" — PaddleOCR's Architecture
           puts the classification layers under a top-level `head` module, so this
           avoids touching backbone/neck tensors that might coincidentally share a
           dimension size)
        2) one of its axes has length == old_dict_len + k for some small k in `offsets`
           (k absorbs special tokens like CTC's blank or NRTR's bos/eos, whose exact
           count can vary and isn't worth hardcoding)
      """
      rng = np.random.default_rng(seed)
      state = paddle.load(old_ckpt_path)
      new_state, resized_log, skipped_log = {}, [], []

      for key, tensor in state.items():
          arr = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
          shape = arr.shape
          match_axis, match_k = None, None
          for axis, size in enumerate(shape):
              for k in offsets:
                  if size == old_dict_len + k:
                      match_axis, match_k = axis, k
                      break
              if match_axis is not None:
                  break

          if match_axis is not None and name_filter in key:
              new_shape = list(shape)
              new_shape[match_axis] = new_dict_len + match_k
              new_arr = rng.normal(0, init_std, size=new_shape).astype(arr.dtype)
              old_slices = tuple(slice(0, s) if a == match_axis else slice(None) for a, s in enumerate(shape))
              new_arr[old_slices] = arr          # old weights preserved exactly
              new_state[key] = paddle.to_tensor(new_arr)
              resized_log.append((key, shape, tuple(new_shape)))
          else:
              new_state[key] = tensor
              if match_axis is not None:          # size matched but name didn't — review manually
                  skipped_log.append((key, shape))

      paddle.save(new_state, new_ckpt_path)
      return resized_log, skipped_log
  ```

  Usage:

  ```python
  # 1) build the extended dictionary — APPEND, don't reorder, so existing
  #    class indices (and their pretrained weights) stay aligned
  with open("PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt", encoding="utf-8") as f:
      old_dict = [l.rstrip("\n") for l in f]
  extended_dict = old_dict + missing_chars   # from the Phase 0 coverage check
  with open("ppocrv6_dict_jp_extended.txt", "w", encoding="utf-8") as f:
      f.write("\n".join(extended_dict) + "\n")

  # 2) resize the pretrained checkpoint to match
  resized, skipped = resize_vocab_checkpoint(
      "PP-OCRv6_small_rec_pretrained.pdparams", "PP-OCRv6_small_rec_pretrained_resized.pdparams",
      old_dict_len=len(old_dict), new_dict_len=len(extended_dict),
  )
  print("resized:", resized)     # confirm these look like the head's classifier/embedding tensors
  print("check manually:", skipped)  # anything here matched by size only — inspect before trusting
  ```

  Then in the yml, point `Global.character_dict_path` at `ppocrv6_dict_jp_extended.txt` and `Global.pretrained_model` at `PP-OCRv6_small_rec_pretrained_resized.pdparams` instead of the original. When training starts, PaddleOCR should report a **clean load with no shape-mismatch warnings** for the head — that's your confirmation the resize actually lined up with the real model definition. I validated the resize mechanics (shape growth, exact preservation of old weights, correct rejection of same-sized-but-unrelated tensors) against synthetic tensors shaped like a CTC+NRTR head before including this — but I don't have your actual checkpoint, so treat the printed `resized`/`skipped` lists as a required sanity check, not a formality.

> **Note on the dictionary filename:** PaddleOCR has followed a `ppocr{version}_dict.txt`-style naming convention for a few releases now (e.g. v5 used `ppocr/utils/dict/ppocrv5_dict.txt`), so PP-OCRv6's is very likely `ppocrv6_dict.txt` in the same directory — but confirm the exact path by opening `Global.character_dict_path` in the actual yml file once you've cloned the repo in Phase 2, rather than trusting this guess blindly.

---

## Phase 1 — Get your data into Kaggle

With 400k images, don't upload through the notebook file browser — create a proper **Kaggle Dataset**:

1. Zip the whole `paddle_ocr_data/` folder (images + both label files) into one archive. Kaggle's dataset upload UI auto-extracts a single zip, and it sidesteps the "50 top-level files" limit that a raw folder of hundreds of thousands of loose images would blow through.
2. On kaggle.com → **Datasets → New Dataset**, upload the zip. Kaggle's current per-dataset cap is 200GB, so 400k small recognition crops should fit comfortably.
3. Keep it **private** unless you want it public.

You'll attach this dataset to your training notebook as a read-only input mounted at `/kaggle/input/<your-dataset-slug>/...`.

---

## Phase 2 — Notebook environment

New Notebook → **Settings** (right sidebar):
- **Accelerator:** `GPU P100` or `GPU T4 x2`. T4 x2 gives you two 16GB GPUs and lets you use distributed training to roughly halve wall-clock time; P100 is a single, slightly older 16GB GPU. Either works — pick T4 x2 if you want to use both cards.
- **Internet:** ON — you need it to `git clone` and install packages.
- If you've never used a GPU accelerator on your account, Kaggle requires phone verification first.
- Add your dataset from Phase 1 via **Add Input**.

Then in a notebook cell:

```bash
!nvidia-smi   # confirm you actually got a GPU and see which one

!git clone https://github.com/PaddlePaddle/PaddleOCR.git
%cd PaddleOCR
!git checkout v3.7.0    # the release that introduced PP-OCRv6 — pin it for reproducibility;
                        # check the Releases page for anything newer before you commit to this

# PaddlePaddle GPU wheel — cu126 index matches Kaggle's current CUDA 12.x images
!python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!pip install -r requirements.txt
```

A couple of environment gotchas that specifically bit a recent PP-OCRv6_small_rec fine-tune on Kaggle's free T4s, worth pre-empting:

- **`opencv` import errors** (`libGL.so.1` missing) are common on minimal Linux images: `!apt-get update && apt-get install -y libgl1` fixes it.
- **NumPy ≥ 2.0 can crash AMP training** — there's a known incompatibility where Paddle's AMP loss-scale NaN/inf check does a `float()` conversion on a NumPy array in a way that raises under NumPy 2.x. Kaggle images currently ship NumPy 2.x by default. If you plan to train with AMP (recommended — see Phase 3) and hit a crash mid-training pointing at the loss-scaler, pin `numpy<2`:
  ```bash
  !pip install "numpy<2"
  ```
  Do this as a preventive step if you're using AMP, so you don't lose GPU-hours to a mid-run crash.
- Do **not** additionally `pip install paddleocr` (the PyPI inference package) in this environment — it pulls in `torch`/`transformers` and can conflict with the Paddle install you actually need for `tools/train.py`. Training only needs the cloned repo plus `paddlepaddle-gpu`.

---

## Phase 3 — Configure the yml

The config you want already exists in the repo you cloned:

```bash
!cp configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml
```

Download the pretrained (Japanese-capable) recognition weights:

```bash
!wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
```

Open `configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml` and edit these fields (leave `Architecture`, `Loss`, and `PostProcess` structure alone — they have to match the pretrained checkpoint):

| Section | Field | Set to | Why |
|---|---|---|---|
| `Global` | `pretrained_model` | `./PP-OCRv6_small_rec_pretrained` (see extension note below) | warm-start from the Japanese-capable checkpoint |
| `Global` | `save_model_dir` | `/kaggle/working/output/ppocrv6_small_jp_ft` | `/kaggle/input` is read-only; only `/kaggle/working` persists to a commit |
| `Global` | `character_dict_path` | leave as shipped, unless Phase 0's coverage check said otherwise | Japanese is already in the default dict |
| `Global` | `max_text_length` | raise from the default (usually 25) if Phase 0 found longer labels | must cover your longest label |
| `Global` | `epoch_num` | start with something like `10–15` | see epoch-planning note below |
| `Global` | `save_epoch_step` | `1` | keep a checkpoint every epoch so you can pick the best one later, not just the last |
| `Global` | `eval_batch_step` *(or `eval_epoch_step`, whichever key is actually present in this yml)* | tuned so eval runs roughly once per epoch, e.g. `[0, 2000]` for a few-thousand-step epoch | frequent enough to track progress, not so frequent it slows training |
| `Global` | `use_amp` | `true` | mixed precision roughly doubles throughput on T4/P100 |
| `Optimizer` → `lr` | `learning_rate` | reduce from the shipped default (commonly `0.0005`) to something like `1e-4`–`2e-4` | you're fine-tuning an already-good model, not training from scratch — a gentler LR avoids wrecking the pretrained representations |
| `Train` → `dataset` | `data_dir` | `/kaggle/input/<your-dataset-slug>/paddle_ocr_data` | root that `images/...` paths are relative to |
| `Train` → `dataset` | `label_file_list` | `[/kaggle/input/<your-dataset-slug>/paddle_ocr_data/train_label.txt]` | |
| `Train` → `dataset` → `sampler` → `first_bs` | start around `128`–`192`, adjust to avoid OOM | small tier is light; T4/P100 have 16GB, so you likely have headroom above the config's shipped default — but see the note below |
| `Eval` → `dataset` | `data_dir` / `label_file_list` | same pattern, pointing at `val_label.txt` | |

> **On `batch_size_per_card: *bs`:** if your yml shows this instead of a plain number, it's a YAML alias, not a literal — `*bs` reuses whatever value is defined elsewhere in the file with a matching `&bs` anchor. In the sibling `PP-OCRv5_server_rec.yml` (same repo, fetched directly to confirm), that anchor sits at `Train.dataset.sampler.first_bs: &bs 128`, inside a `MultiScaleSampler` that trains across a few width×height buckets (e.g. `scales: [[320,32],[320,48],[320,64]]`) and — since `fix_bs: false` — scales the *effective* batch down for the larger buckets via `divided_factor`, so memory use stays roughly constant across scales. PP-OCRv6_small_rec almost certainly mirrors this. **Edit the `first_bs:` line itself** (keep the `&bs` tag attached) — editing `batch_size_per_card` does nothing, it's just a pointer. When judging whether a batch size will OOM, think about the *largest* scale bucket, not just `first_bs` in isolation. Also note: in that same sibling config, `Eval.loader.batch_size_per_card` was a separate plain number, not linked to `*bs` — change it independently if you want.

> **On the `.pdparams` extension:** PaddleOCR's own docs are inconsistent about whether `Global.pretrained_model`/`Global.checkpoints` should include the `.pdparams` suffix — some official examples include it, some real-world write-ups explicitly say to omit it. If training fails to find your weights immediately, try toggling this (add or drop the suffix) before assuming something else is wrong.

`character_dict_path` is normally defined once in `Global` via a YAML anchor and referenced elsewhere (e.g. in `PostProcess`) — so you only need to change it in that one place if you do end up needing a custom dictionary.

**On epoch count:** with 400k samples you're getting a lot of gradient signal per epoch already, and you're warm-starting from a checkpoint that can already read Japanese — this is not the 15-epoch cold-start situation a from-scratch language would need. Start smaller (10–15 epochs), watch the eval accuracy curve, and extend only if it's still climbing. `save_epoch_step: 1` means you can always go back and pick whichever epoch actually had the best validation accuracy, rather than assuming the last one is best (a repetitive document domain like addresses/company names can start overfitting well before your last epoch).

---

## Phase 4 — Smoke test before you commit real GPU hours

Kaggle's weekly GPU quota is limited (currently on the order of 30 hours/week), so don't discover a config typo three hours into a run. Point `label_file_list` at a 500–1000 line slice of your training file and run for a couple hundred steps first:

```bash
!head -n 1000 /kaggle/input/<your-dataset-slug>/paddle_ocr_data/train_label.txt > /kaggle/working/smoke_train.txt
!head -n 200  /kaggle/input/<your-dataset-slug>/paddle_ocr_data/val_label.txt   > /kaggle/working/smoke_val.txt
```

Temporarily repoint `label_file_list` at these two files, run a few hundred steps, confirm loss is decreasing and no shape/dictionary errors appear, then switch back to the full files for the real run.

---

## Phase 5 — Launch training (and survive the 12-hour wall)

Single GPU (P100 or one T4):

```bash
!python3 tools/train.py -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml \
    -o Global.pretrained_model=./PP-OCRv6_small_rec_pretrained
```

Two GPUs (T4 x2):

```bash
!python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py \
    -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml \
    -o Global.pretrained_model=./PP-OCRv6_small_rec_pretrained
```

**Estimate your own throughput** rather than trusting a generic number: `tools/train.py` logs speed every `print_batch_step` iterations. Watch the first couple hundred steps, compute steps/epoch (≈ `400000 / effective_batch_size`), and multiply out to get a real epoch-time estimate for your actual hardware and image sizes — inference-speed benchmarks published for PP-OCRv6_small_rec don't include the backward pass or your data loader, so they won't directly tell you training throughput.

**Kaggle sessions are capped** — recent reports put the GPU session limit around 12 hours (this has moved between roughly 9 and 12 hours over time, so treat it as approximate and check your notebook's current limit), with a weekly quota on top. For 400k samples you should plan for **multiple sessions** almost regardless of your throughput estimate. To make that painless:

1. Always run long training as a **committed** run (**Save Version → Save & Run All**), not just an interactive session — commits keep running on Kaggle's infrastructure even if you close the tab, and are the standard way to do unattended long runs.
2. When a session is about to end (or has ended), the commit's `/kaggle/working` output becomes downloadable as that notebook version's **Output**. Start your next notebook version, add your *own previous notebook version* as a data source (**Add Data → Notebook Output Files**), which mounts its `output/ppocrv6_small_jp_ft/` checkpoints read-only.
3. Resume with `Global.checkpoints`, **not** `Global.pretrained_model` — `checkpoints` restores the optimizer state and epoch count too, so the LR schedule continues smoothly instead of resetting:
   ```bash
   !python3 tools/train.py -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml \
       -o Global.checkpoints=/kaggle/input/<your-previous-notebook-output>/output/ppocrv6_small_jp_ft/latest
   ```
4. Keep `epoch_num` **fixed across all sessions** (don't bump it up mid-run) so the cosine LR schedule you configured stays consistent from session to session — only the `checkpoints` path changes.
5. If you're using `use_amp` with EMA also enabled somewhere in the config, be aware there's a known dtype conflict (EMA's fp32 shadow weights vs. the AMP fp16 model) that can break evaluation — turning EMA off is the simpler fix, and your epoch-by-epoch checkpoint selection already gives you the "pick the best point" benefit EMA is usually there for.

---

## Phase 6 — Evaluate

```bash
!python3 tools/eval.py -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml \
    -o Global.pretrained_model=/kaggle/working/output/ppocrv6_small_jp_ft/best_accuracy
```

Compare a few checkpoints (`best_accuracy` vs. specific epoch snapshots) rather than assuming later = better, especially if you saw signs of overfitting in the loss curves.

---

## Phase 7 — Export for inference

```bash
!python3 tools/export_model.py -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_jp_ft.yml \
    -o Global.pretrained_model=/kaggle/working/output/ppocrv6_small_jp_ft/best_accuracy \
       Global.save_inference_dir="/kaggle/working/ppocrv6_small_jp_ft_infer/"
```

This produces `inference.json`, `inference.pdiparams`, and `inference.yml` — the deployable format PaddleOCR's `TextRecognition`/`PaddleOCR` API loads directly via `model_dir=`. Sanity-check on a handful of real held-out crops before considering the job done:

```python
from paddleocr import TextRecognition
model = TextRecognition(model_dir="/kaggle/working/ppocrv6_small_jp_ft_infer/")
for res in model.predict(input="some_val_crop.jpg", batch_size=1):
    res.print()
```

(If you followed the "don't pip install paddleocr" advice for training, it's fine to `pip install paddleocr` at this final inference-testing step — that concern was specifically about it interfering with the training environment.)

---

## Troubleshooting cheat-sheet

- **Shape mismatch loading the pretrained head** → you edited `character_dict_path` to a different dictionary size than the checkpoint expects. Only do this if Phase 0's coverage check justified it, and expect the head weights to load non-strictly (randomly initialized for the new rows).
- **`.pdparams` suffix confusion** → see the callout in Phase 3; try both with and without.
- **OOM** → lower `batch_size_per_card` first; `use_amp: true` also reduces memory pressure.
- **AMP crash referencing NaN/inf loss scale** → NumPy ≥2 incompatibility, pin `numpy<2` (Phase 2).
- **`cv2`/`libGL` import error** → `apt-get install -y libgl1` (Phase 2).
- **Training loss fine but eval accuracy stuck near zero** → almost always a dictionary/decoding mismatch (wrong `character_dict_path`, or checking predictions against the wrong postprocess config) rather than a training problem — check this before touching hyperparameters.
- **Session ends before you expected** → this is normal for a 400k-sample run; the resume workflow in Phase 5 is the intended pattern, not a fallback for something going wrong.

---

## Sources

- [PP-OCRv6 Introduction — PaddleOCR docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- [Text Recognition Module Tutorial — PaddleOCR docs](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html) (dataset format, training/eval/export commands, model download links)
- [PaddlePaddle/PaddleOCR GitHub releases — v3.7.0](https://github.com/PaddlePaddle/PaddleOCR/releases) (PP-OCRv6 launch)
- [PaddlePaddle/PP-OCRv6_small_rec — Hugging Face model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec)
- [medyas/arabic_PP-OCRv6_small_rec — Hugging Face model card](https://huggingface.co/medyas/arabic_PP-OCRv6_small_rec) (real-world Kaggle T4×2 fine-tuning run: session limits, AMP/EMA/NumPy gotchas, checkpoint-resume pattern)
- [Kaggle Datasets technical specifications](https://www.kaggle.com/docs/datasets) (200GB upload limit, 50 top-level file limit)
