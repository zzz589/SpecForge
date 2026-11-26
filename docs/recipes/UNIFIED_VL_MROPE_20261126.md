# Unified single-image training on hn47

The canonical training source is `xj/src/Specforge-pr785/SpecForge-pr785`.
Use `xj/envs/specforge785-dflash2-vl/bin/python` and explicit PYTHONPATH entries
for this repository and `xj/src/sglang-dflash2-vl-mrope/python`.
`commands/train_vl_unified.sh CONFIG.yaml --role auto` sets these imports and
checks their actual origins. Select completely idle GPUs before a launch.

The pinned SGLang source does not itself contain PR785's spec-capture sink.
The wrapper builds a verified training-only view using
`scripts/prepare_sglang_capture_overlay.py` and
`patches/sglang/hn47-01889693/`. Unmodified files link to the pinned source;
capture-patch targets are copied and patched in a separate cache. The base
commit, existing DSpark port and all base file hashes must match before use.
The pinned SGLang directory is never edited. Each view records the actual
patched-file hashes in `capture-overlay.json`. Ordinary serving continues to
use its original PYTHONPATH; this capture view is a training dependency, not
a new performance baseline. Direct CLI launches need this view on PYTHONPATH;
use the wrapper to avoid accidentally importing an unpatched SGLang package.

| Algorithm | Strategy | Architecture | Configuration |
|---|---|---|---|
| DFlash-VL | `dflash` | `DFlashDraftModel` | `configs/qwen3-vl-4b-dflash-mrope.json` |
| DFlash2-VL | `dflash` | `DFlash2DraftModel` | Existing B9 mRoPE checkpoints in the migrated recipes; `configs/qwen3-vl-4b-dflash2-mrope-b16.json` for a new B16 draft |
| DSpark-VL | `dspark` | `DSparkDraftModel` | `configs/qwen3-vl-4b-dspark-mrope.json` |

All three share the single-image Qwen3-VL input/capture path, interleaved mRoPE,
trainer and export infrastructure. Their objectives, heads, target layers,
block sizes and rotary theta remain algorithm/checkpoint-specific. In particular,
the current DFlash/DFlash2 theta is 5,000,000 and DSpark theta is 1,000,000.
Do not substitute B16 configuration when loading B9 weights.

## DFlash2 recipes

Under `examples/configs/online/disaggregated/managed-local/`:

- `qwen3-vl-4b-dflash2-mrope-original2272.yaml`: aligned original-image inputs,
  saved response token IDs, B9 full attention, D-PACE plus selector.
- `qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed40031-dsparkmatched.yaml`:
  raw JSONL, original constant-lambda loss and selector settings.
- `qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed41561.yaml`: historical mapped
  cache entry. Its manifest references missing external cache files on hn47
  as of 2026-11-26; preflight fails explicitly until those files are restored.
- `qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed41561-raw.yaml`: raw JSONL
  alternative with the same model/objective settings. Eligibility is recomputed;
  its usable count must not be called 41,561 without checking preparation output.

Migrated recipes use fresh output/control/cache namespaces. Original independent
directories, checkpoints and historical experiment outputs remain available as
provenance; future training imports this repository.

## Input and position contract

The validated scope is one image/question followed by one assistant answer.
Structured message parts retain the image's exact position. Ordinary JSONL uses
the target tokenizer/processor. Records containing `generation_prompt_ids` and
`response_token_ids` preserve the original token IDs and verify the current
template prefix and decoded answer; they must be single-turn Qwen records.
Optional `data.exact_journal_path` also checks the complete generation journal,
including image bytes. A mismatch is fatal, never a silent skip or re-tokenization.
Visual-score journal helpers are diagnostics; they do not add a training weight.

`data.mapped_manifest_path` retains cached tokens without Dataset.map. It verifies
each shard's row count, the deterministic source permutation, collapsed source
tokens and masks. At row access it checks the actual target-processor image grid
and sends the same validated image bytes. Equal image-token counts are insufficient:
different height/width layouts require different mRoPE coordinates.

Every path keeps EOS as a label and masks the template suffix after EOS. Cached
eligibility is filtered before the scheduler sees the sample count. The capture
server must use `SGLANG_MM_AVOID_RETOKENIZE=1`; the managed launcher sets it.
`int64 [B,S,3]` positions travel with features, become `[3,B,S]` for rotary, and
rotate both context K and draft Q/K. Candidate text positions continue from each
anchor on all axes, including tail slots past the recorded response. Attention
and caches keep logical token indices, not compressed rotary coordinates.

DFlash/DFlash2 predict token `anchor+k` at slot k and exclude slot 0 from loss.
DSpark predicts token `anchor+k+1`. The shared trainer retains pooled gradient
normalization and forward/backward no_sync handling. DSpark's confidence-head
disablement and next-token sampling fixes are preserved.

## Checkpoints and validation

Full resume verifies architecture/compute config, mRoPE convention, theta,
selector settings, EOS masking and optimizer-step objective normalization.
An older full checkpoint without these contracts is not silently resumed;
use a new run with compatible draft weights when provenance is established.
Changing rotary configuration does not retroactively make old weights mRoPE-trained.

Run `python -m unittest` with the modules in `tests/unified_vl_suite.py`, or run
that suite directly. `tests/probes/unified_vl_probe.py` additionally exercises
the installed Qwen3-VL tokenizer/processor, selected real SGLang processor methods,
tiny actual Qwen3-VL models, three draft architectures, exact tokens, Arrow data,
FP32/BF16 backward and export/reload. This CPU probe does not itself validate
full-size CUDA training or live Mooncake capture; separate GPU results are recorded
under the dated merge audit directory.
