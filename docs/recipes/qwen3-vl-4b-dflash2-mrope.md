# Qwen3-VL-4B DFlash2 training with interleaved mRoPE

This branch adds the missing Qwen3-VL position semantics to the PR785 DFlash2
path. It keeps the text path backward compatible and requires `position_ids`
only for streaming multimodal runs.

## What is implemented

- Qwen3-VL temporal/height/width position construction, matching
  `Qwen3VLModel.get_rope_index`.
- `[B, S, 3]` position passthrough through SGLang capture and Mooncake.
- DFlash-family anchor/block position gathering for three-axis positions.
- Interleaved Qwen3-VL rotary embedding in `DFlash2DraftModel`.
- Optional reuse of mapped `image_grid_thw`; no `Dataset.map` rerun.
- Auditable text-DFlash2 to Qwen3-VL init adaptation with bitwise unchanged
  draft tensors.
- The public DFlash2 objective `0.1 CE + 0.9 TV`, represented exactly by
  `lk_loss_type=lambda`, `kl_scale=0.1`, `kl_decay=0`.

## Prepared artifacts

- Init checkpoint:
  `/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1`
- 100-sample smoke recipe:
  `examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-textvqa100-smoke.yaml`
- Four-GPU 100-sample validation recipe:
  `examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-textvqa100-4gpu.yaml`
- DSpark-matched comparison recipe (40,031 effective samples from the 41,561-row source):
  `examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed40031-dsparkmatched.yaml`
- Full 41,561-row mapped-cache recipe (not directly comparable to the existing DSpark run):
  `examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed41561.yaml`

## Run

```bash
export DFLASH2_ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
export DFLASH2_REPO="$DFLASH2_ROOT/src/SpecForge-pr785-dflash2-vl-mrope"
export DFLASH2_ENV="$DFLASH2_ROOT/envs/specforge785-dflash2-vl"
export PYTHONPATH="$DFLASH2_REPO"
export PATH="$DFLASH2_ENV/bin:$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
cd "$DFLASH2_REPO"

# Smoke test: GPU 7 trainer, GPU 5 capture server.
"$DFLASH2_ENV/bin/python" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-textvqa100-smoke.yaml \
  --role auto

# Validated four-GPU run: GPUs 6-7 trainer, GPUs 2 and 5 capture.
"$DFLASH2_ENV/bin/python" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-textvqa100-4gpu.yaml \
  --role auto

# DSpark-matched comparison: GPUs 6-7 trainer, GPUs 2 and 5 capture.
"$DFLASH2_ENV/bin/python" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed40031-dsparkmatched.yaml \
  --role auto
```

The comparable run must print `VLM prompt preparation done: 40031 prepared,
1530 skipped`. It matches the completed DSpark run at 3 epochs, 32 anchors,
global batch size 4, learning rate `1e-5`, cosine scheduling, warmup `0.03`, and
seed `20261014`. The public text-only settings (`512` anchors, `6e-4`, one
epoch) are intentionally not used for this VL comparison. DFlash2 keeps its
method-specific `0.1 CE + 0.9 TV` objective.

The comparison path does not invoke Hugging Face `Dataset.map`, but it does
repeat runtime VLM prompt preparation to reproduce DSpark's exact filtering.
The separate `mixed41561.yaml` path reuses all mapped rows and must print
`MAPPED VLM CACHE HIT`, `rows=41561`, `Dataset.map skipped`, and
`cached image_grid_thw reused`; because it includes the 1,530 rows skipped by
the existing DSpark run, it is not an exact comparison.

## Validation already completed

- Existing DFlash2 suite: 33 tests passed before adding the mRoPE model test.
- Qwen3-VL positions: exact elementwise equality with the Transformers target.
- Init checkpoint: all 81 draft keys loaded, no missing keys; source/output
  weight SHA-256 values are identical.
- Real DFlash2 B9 forward/backward on GPU: finite gradients with
  `position_ids.shape == [1, 32, 3]`.
- Mapped cache sample: 407 input tokens and 407 position triplets.
- Four-GPU managed-local run: 100/100 samples produced, rank 0 and rank 1 each
  consumed 50, global step 50 saved successfully, no training exception, and
  all managed services released their ports and GPU allocations.
