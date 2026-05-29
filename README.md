# Ndizi Swahili — w2v-BERT CTC bundle

Self-contained layout: **`config_files/w2vbert/`** (Hub CTC / w2v-BERT), **`config_files/whisper/`** (Whisper), **`src/`** (data, model factory, training), **`scripts/train_model.py`** (CTC training), **`scripts/train_whisper.py`** (Whisper), and **`scripts/evaluate_asr_batch.py`** (batched test evaluation).

## Prerequisites

- Python with **torch**, **transformers**, **datasets**, **evaluate**, **tqdm** (see [`../requirements.txt`](../requirements.txt) when this folder lives under `May_4_experiment/`).
- Optional: **PyYAML** if you use `.yaml` configs (`.json` needs only the stdlib).
- GPU recommended.

## Quick start

```bash
cd /path/to/ndizi_mlops
pip install -r ../requirements.txt   # adjust path to your requirements file

chmod +x bash_scripts/run_train_ndizi_w2vbert.sh bash_scripts/run_train_ndizi_whisper.sh bash_scripts/run_eval_asr_batch.sh
./bash_scripts/run_train_ndizi_w2vbert.sh
# Whisper training (optional config path as first argument):
# ./bash_scripts/run_train_ndizi_whisper.sh
# Batched eval (CTC or Whisper; pass through all evaluate_asr_batch.py flags):
# ./bash_scripts/run_eval_asr_batch.sh --model_path ... --output_dir eval/run1 --test_datasets smutuvi/ndizi-1:test

# Or directly:
python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json

# Drop clips longer than 30s for this run (overrides config max_input_seconds):
python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged_10epoch.json --max-input-seconds 30

# Keep all lengths even if the config caps duration:
python3 scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json --no-max-input-filter
```

`bash_scripts/run_train_ndizi_w2vbert.sh` sets cache env vars then invokes `scripts/train_model.py`. **`bash_scripts/run_train_ndizi_whisper.sh`** does the same for **`scripts/train_whisper.py`**. **`bash_scripts/run_eval_asr_batch.sh`** forwards all arguments to **`scripts/evaluate_asr_batch.py`** (works for CTC and Whisper; use **`--backend auto`** by default so Whisper runs are detected from `training_config_resolved.json`).

## Evaluation (batched Hub test splits)

[`scripts/evaluate_asr_batch.py`](scripts/evaluate_asr_batch.py) loads your saved checkpoint, decodes in batches, and writes **`metrics.json`** (per-split + **pooled** WER/CER) and **`predictions.json`** (one record per utterance). Mono ASR only (no language-ID head). Text cleaning and WER reference formatting follow **`train_model.py`** when **`training_config_resolved.json`** sits next to the checkpoint (or pass **`--training_config`**). By default there is **no** max-audio cap (same idea as the upstream mono batch eval); pass **`--max_audio_seconds 30`** to skip clips longer than training **`max_input_seconds`**.

```bash
cd /path/to/ndizi_mlops

python3 scripts/evaluate_asr_batch.py \
  --model_path inprogress/ndizi-w2vbert-merged-1epoch-w2vbert20/facebook-w2v-bert-2.0-DDMMYYYY-HHMMSS \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/my_run_test \
  --batch_size 8 \
  --max_audio_seconds 30
```

Use **`--processor_path`** only if `AutoProcessor.from_pretrained(model_path)` fails (then point at the directory that contains `preprocessor_config.json` + tokenizer next to your `ctc_tokenizer/`).

**CUDA memory:** decoding streams features per batch (no full-split precompute). If you still OOM on long clips, try **`--chunk_long_audio_seconds 30`** (chunked greedy decode, preds joined with spaces), **`--fp16`**, smaller **`--batch_size`**, or **`--max_audio_seconds 30`**. You can also set **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce fragmentation. The previous “precompute every row” driver is kept as [`scripts/evaluate_asr_batch_backup_precompute_all_rows.py`](scripts/evaluate_asr_batch_backup_precompute_all_rows.py) for tiny splits or comparison only.

For CSV output, chunking, or QC filters, you can still use [`../evaluate_w2v_bert_ctc.py`](../evaluate_w2v_bert_ctc.py) from `May_4_experiment/` with the same `--model_id`.

## `src/` layout (what was added)

| Module | Role |
|--------|------|
| `src/utils/config.py` | `ASRConfig` dataclass + JSON/YAML `load_config` |
| `src/utils/cache.py` | Stubs for optional encoded-dataset caching |
| `src/data/dataset.py` | `load_datasets` (merged Hub ids or single `dataset_path`), `load_hub_processor`, optional custom `create_processor` / `build_vocabulary` for non–Hub-CTC bases |
| `src/data/preprocessing.py` | Text cleaning or identity pass for Hub CTC |
| `src/data/dataset_encoders.py` | `ASRDatasetEncoder` → model inputs + `labels` |
| `src/models/factory.py` | `create_asr_model` (Hub CTC) and `create_asr_model_for_custom_vocab` (e.g. `w2v-bert-2.0` + built vocab) |
| `src/training/collator.py` | `DataCollatorCTCWithPadding` |
| `src/training/metrics.py` | WER/CER + optional prediction JSON dumps |
| `src/training/trainer.py` | `TrainingArguments` + `Trainer` wiring |

## Config

Configs live under **`config_files/w2vbert/`** (Hub CTC / w2v-BERT; use with `scripts/train_model.py`) and **`config_files/whisper/`** (set `"stack": "whisper"`; use with `scripts/train_whisper.py`). Each w2v-BERT experiment stem has a **`.json`** and matching **`.yaml`** in `w2vbert/`.

| Path (under `config_files/w2vbert/`) | Purpose |
|------|---------|
| `ndizi_w2vbert_merged` | `smutuvi/ndizi-1` + `smutuvi/ndizi-1-2025`, pooled `validation` |
| `ndizi_w2vbert_ndizi1_only` | `smutuvi/ndizi-1` only |
| `ndizi_w2vbert_2025_only` | `smutuvi/ndizi-1-2025` only |
| `ndizi_w2vbert_merged_alt_hparams` | Merged data; larger batch, step-based eval |
| `ndizi_w2vbert_merged_1epoch` | Merged data, **1 epoch**, **`facebook/w2v-bert-2.0`** + custom char vocab (`use_hub_ctc_checkpoint: false`) |
| `ndizi_w2vbert_merged_10epoch` | Merged w2v-bert-2.0, **10 epochs**; **`max_input_seconds: null`** keeps all clip lengths—use **`--max-input-seconds 30`** when you want a 30s cap without editing the file |

Whisper: see `config_files/whisper/ndizi_whisper_small_merged.json` (example merged Hub setup).

### Long audio

There is **no** train-time audio chunking; long clips stay as single examples when **`max_input_seconds`** is **`null`**. Prefer the knobs above (and CLI duration overrides) over ad‑hoc chunking.

Important keys:

- **`max_input_seconds`** (default `30`): rows with **longer** `audio_duration` are **dropped** from train and eval. Set to **`null`** to **keep all lengths** (long clips). For VRAM, prefer smaller **`batch_size`**, **`gradient_checkpointing: true`**, **`fp16: true`**, lower **`per_device_eval_batch_size`**, and higher **`gradient_accumulation_steps`** to preserve effective batch size. **`train_model.py`** can override duration filtering without editing the file: **`--max-input-seconds 30`** (drop clips over 30s) or **`--no-max-input-filter`** (keep all lengths).
- **`use_hub_ctc_checkpoint`** (default `true`): use `AutoProcessor` / `AutoModelForCTC` from **`pretrained_model`** (e.g. `facebook/wav2vec2-bert-rel-pos-large`).
- **`use_hub_ctc_checkpoint`: `false`** with **`pretrained_model`: `facebook/w2v-bert-2.0`**: builds CTC **`vocab.json`** from **`character_set`**, `Wav2Vec2BertProcessor` + `SeamlessM4TFeatureExtractor`, and loads the backbone with a **resized CTC head**. Example: `ndizi_w2vbert_merged_1epoch.json`. Expand **`character_set`** if labels contain characters that get removed by cleaning.

Copy `.env.example` to `.env` for optional `HF_TOKEN` / `HF_API_KEY` and cache paths.

## Model choice

- **`facebook/wav2vec2-bert-rel-pos-large`** with **`use_hub_ctc_checkpoint: true`** (default merged configs).
- **`facebook/w2v-bert-2.0`** with **`use_hub_ctc_checkpoint: false`** and a tuned **`character_set`** (see `ndizi_w2vbert_merged_1epoch` configs). For heavier custom pipelines you can still use [`../train_w2v_bert_ctc.py`](../train_w2v_bert_ctc.py).
- **`../ndizi_finetune_w2vbert.py`** remains an older standalone script; new work should prefer this bundle’s `scripts/train_model.py` + `src/`.

## Outputs

Checkpoints and processor are written under **`output_dir/<experiment_name>/`** where `experiment_name` includes a timestamp (see `ASRConfig.get_experiment_name`). `metrics.json` and `training_config_resolved.json` are saved there after training.

## Optional: HTCondor / Docker

See `run.sub.example`. Point the job at `bash_scripts/run_train_ndizi_w2vbert.sh` and a GPU image with the same Python dependencies.

## Compared to a full multi-thousand-line ASR repo

Not ported (can be added later): Weights & Biases reporting, DDP launch helpers, on-disk **encoded dataset** cache, LID heads / collators, exhaustive data QA filters, and the full “every ablation flag” surface. This bundle focuses on **merged Ndizi Hub data** with either **Hub CTC fine-tune** or **`w2v-bert-2.0` + char vocab**, with extension points in `src/data/dataset.py` and `src/utils/config.py`.
