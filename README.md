# ndizi_mlops

Training and evaluation toolkit for **Ndizi Swahili ASR** on Hugging Face datasets ([`smutuvi/ndizi-1`](https://huggingface.co/datasets/smutuvi/ndizi-1), [`smutuvi/ndizi-1-2025`](https://huggingface.co/datasets/smutuvi/ndizi-1-2025)). Supports two stacks:

| Stack | Entry script | Configs |
|-------|--------------|---------|
| **w2v-BERT CTC** | `scripts/train_model.py` | `config_files/w2vbert/` |
| **Whisper** | `scripts/train_whisper.py` | `config_files/whisper/` |

Batched test-set evaluation: `scripts/evaluate_asr_batch.py` (CTC and Whisper). Past runs are under `eval/`.

## Setup

```bash
git clone https://github.com/smutuvi/ndizi_mlops.git
cd ndizi_mlops
pip install torch transformers datasets evaluate tqdm
# Optional: PyYAML for .yaml configs (.json needs only stdlib)
```

Copy secrets locally (not in git):

```bash
# .env — HF_TOKEN, cache dirs, etc.
export HF_TOKEN=...
```

GPU recommended.

## Train

**w2v-BERT (Hub CTC or custom char vocab on w2v-bert-2.0):**

```bash
./bash_scripts/run_train_ndizi_w2vbert.sh
# or
python scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json
```

**Whisper:**

```bash
./bash_scripts/run_train_ndizi_whisper.sh config_files/whisper/ndizi_whisper_large_v3_turbo_merged.json
```

**Duration filter (CLI overrides config):**

```bash
python scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged_10epoch.json --max-input-seconds 30
python scripts/train_model.py --config config_files/w2vbert/ndizi_w2vbert_merged.json --no-max-input-filter
```

Checkpoints land in `output_dir/<experiment_name>/` with `training_config_resolved.json` and `metrics.json`. Local weights are gitignored under `inprogress/`.

## Evaluate

```bash
python scripts/evaluate_asr_batch.py \
  --model_path inprogress/your-run/checkpoint-best \
  --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test \
  --output_dir eval/my_run \
  --batch_size 8 \
  --max_audio_seconds 30
```

Writes **`metrics.json`** (per-split + pooled WER/CER) and **`predictions.json`**. Uses `training_config_resolved.json` next to the checkpoint for text normalization (or pass `--training_config`).

Wrapper: `./bash_scripts/run_eval_asr_batch.sh` (forwards all flags). For OOM on long audio: `--chunk_long_audio_seconds 30`, `--fp16`, smaller `--batch_size`.

Pipeline eval with extra preprocessing: `scripts/evaluate_asr_batch_pipeline.py` and `bash_scripts/run_eval_asr_batch_pipeline.sh` (same reference formatting, post-decode cleanup, and `punct_recall` as `evaluate_asr_batch.py`).

## Repository layout

```
config_files/w2vbert/   # CTC experiment JSON/YAML
config_files/whisper/   # Whisper experiment JSON
scripts/                # train_* and evaluate_* CLIs
src/                    # data loading, QC, models, trainers
bash_scripts/           # thin wrappers (cache env + python)
eval/                   # committed metrics/predictions from past runs
```

| `src/` area | Role |
|-------------|------|
| `data/dataset.py` | Load/merge Hub datasets, processors |
| `data/text_format.py` | Transcript formatting + punctuation recall metrics |
| `data/qc.py` | Optional clip filters (duration, weird-ratio, etc.) |
| `models/factory.py` | Hub CTC or w2v-bert-2.0 + custom vocab |
| `training/trainer.py` | Hugging Face `Trainer` wiring (CTC) |
| `training/whisper_trainer.py` | Whisper fine-tuning |
| `utils/config.py` | `ASRConfig` + JSON/YAML load |

## Config notes

- **`max_input_seconds`**: drop clips longer than N seconds; set `null` when using MMS-FA chunking (`qc_chunk_long_with_mms_fa`).
- **`format_transcripts`** (default `true`): spacing after `.?!`, comma glue fixes via `src/data/text_format.py` before `clean_transcription`.
- **`use_hub_ctc_checkpoint: true`**: Hub CTC tokenizer has **no punctuation**; outputs stay lowercase/run-on. Prefer `false` + `character_set` including `.,?!` for readable CTC (see `ndizi_w2vbert_merged.json`).
- **`use_hub_ctc_checkpoint: false`**: `facebook/w2v-bert-2.0` + custom char vocab from `character_set`.
- **`qc_allow_sentence_punctuation`** (maps to `QCConfig.allow_sentence_punctuation`): `. , ? ! : ;` are not counted as “weird” in QC.
- **`use_formatting_score_for_best`**: checkpoint selection uses composite `score` (WER + CER + `punct_recall`) instead of WER alone.
- Whisper configs set `"stack": "whisper"` and choose model id (e.g. `openai/whisper-large-v3-turbo`).
- Eval: `evaluate_asr_batch.py` / `evaluate_asr_batch_pipeline.py` apply the same reference formatting as training; post-decode formatting is on by default (`--no-format-decode` to disable). Metrics include **`punct_recall`**; with `--normalize jiwer_default`, **`wer`/`cer`** are punctuation-preserving (raw) and **`wer_jiwer`/`cer_jiwer`** (or `wer_normalized`) report jiwer-stripped scores.
- Long-audio chunked decode joins segments with `join_chunk_predictions` (sentence boundary between chunks when needed).

Example w2v-BERT stems: `ndizi_w2vbert_merged`, `ndizi_w2vbert_ndizi1_only`, `ndizi_w2vbert_2025_only`, `ndizi_w2vbert_merged_1epoch`.

## Cluster / Docker

`run.sub.example` shows HTCondor + Docker pointing at `bash_scripts/run_train_ndizi_w2vbert.sh`. Adjust `initialdir` and `executable` to your paths.

## Scope

Focused on **merged Ndizi Hub fine-tuning**, QC-filtered training, and reproducible batched eval—not a full multi-task ASR framework (no LID heads, DDP helpers, or W&B integration in-tree). Extend via `src/data/dataset.py` and `src/utils/config.py`.
