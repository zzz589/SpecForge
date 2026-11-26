# DSpark single-image Q&A with mRoPE

This recipe covers one user image/question and one assistant answer, using a
Qwen3-VL target through the existing server-streaming capture path. The public
offline-v1 provider remains text-only. Video and multi-turn training are outside
the validation scope of this change.

## Configuration

Use the following fields in the existing training configuration; this fragment
is not a complete launch configuration:

```yaml
model:
  input_modality: multimodal
  draft_model_config: configs/qwen3-vl-4b-dspark-mrope.json
training:
  strategy: dspark
```

The new JSON retains the existing B16 DSpark model's five draft layers,
target layers `[1, 9, 17, 25, 33]`, Markov rank 256, confidence head, and
`rope_theta=1000000`. It enables interleaved mRoPE with sections `[24, 20, 20]`.
The old 1-D configurations are unchanged. Multimodal assembly rejects a draft
configuration without explicit mRoPE.

Each JSONL row can use this shape. Prefer an absolute image path; a relative
path is resolved against the training process's working directory:

```json
{"image":"/data/image.png","conversations":[{"role":"user","content":"Describe this image."},{"role":"assistant","content":"A red bicycle."}]}
```

The target tokenizer/processor settings must match the capture server. The
managed multimodal launcher sets `SGLANG_MM_AVOID_RETOKENIZE=1`; keep it enabled
for a separately launched capture server as well.

## Training semantics

1. Preprocessing expands image tokens and produces aligned loss masks and
   `int64 position_ids` with shape `[B, S, 3]`. Capture transports positions,
   selected target features and target final hidden states together.
2. The collator pads all features consistently. The strategy forwards positions
   to the DSpark wrapper; shared block attention rotates context K and draft Q/K
   with the image triplets and text continuation positions.
3. DSpark preserves its next-token objective: slot `k` at anchor `a` predicts
   `input_ids[a+k+1]`, using the teacher hidden state at `a+k`. The Markov head
   receives the corresponding teacher-forced previous token. This differs from
   DFlash's label alignment and must not be changed to the DFlash convention.
4. Supervision includes the first EOS in the assistant span, but excludes the
   template suffix after EOS. Empty answers cannot supply an anchor/next-token
   pair and are filtered before the schedule is planned.
5. CE, distribution L1 and confidence loss contribute additive numerators.
   TrainerCore divides their weighted sum by the pooled token-weight denominator
   over the entire accumulation window and data-parallel group. A locally empty
   microbatch still participates in backward; a globally empty optimizer window
   is rejected before the optimizer update.

## Checkpoints

New checkpoints store `dspark_compute_config`, position convention, mRoPE
sections/theta, EOS-mask semantics and optimizer-step normalization semantics.
Resume rejects missing or changed fields rather than silently changing the
training objective. Export uses DSpark's own checkpoint keys and verifies
configuration and mRoPE provenance.

Old 1-D checkpoints are not mRoPE-trained weights. Legacy 1-D export remains
available with verifiable original-config provenance; this does not authorize
resuming an old checkpoint under the new objective or relabeling it as mRoPE.
Start a new training run with the new configuration for this recipe.

## Validation scope

The repair was checked with CPU regression tests, two-process Gloo gradient
comparison including an empty-rank microbatch, real Qwen3-VL tokenizer/processor
image cases, tiny random Qwen3-VL and DSpark FP32/BF16 forward/backward, export
reload and the actual Trainer save/resume path. CPU BF16 tests explicitly use
the eager loss (`SPECFORGE_DSPARK_FUSED_LOSS=0`); production GPU defaults are not
changed. CUDA/FSDP, Triton loss, live Mooncake capture and full-size target
training remain to be validated in an authorized GPU smoke run.

## Head regressions fixed on 2026-11-26

Setting `training.dspark_confidence_head_alpha=0` now freezes the confidence
head and skips its computation, allowing DDP to run consecutive steps. Other
DSpark parameters, including the Markov head, remain trainable. The default
positive confidence coefficient retains confidence supervision. Checkpoints
from older zero-confidence runs may contain a different optimizer parameter
layout; they are not guaranteed to support full optimizer-state resume.

The internal HF proposal helper now consumes the first B-1 next-token prediction
slots of the last block, matching DSpark training. Tests cover disabled,
vanilla, gated and recurrent Markov heads. This correction does not add an
image-input API to the inherited HF `spec_generate` helper.

Additional CPU checks compare packed single-image blocks with independent and
incremental KV-cache calls using compressed image triplets, including candidate
positions extending past the training sequence. Cache lengths remain logical
token counts rather than compressed mRoPE coordinates.
