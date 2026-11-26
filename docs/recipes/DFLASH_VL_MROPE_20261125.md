# PR785 Qwen3-VL DFlash mRoPE

This local change supports single still-image Qwen3-VL DFlash training. It does
not launch a job, change a checkpoint, update a serving engine, or convert old
1D-trained weights into mRoPE-trained weights.

Use `configs/qwen3-vl-4b-dflash-mrope.json` as `model.draft_model_config`, with
`training.strategy: dflash` and `model.input_modality: multimodal`. The new config
keeps the existing DFlash B16 architecture and target-layer selection, and sets
interleaved sections `[24,20,20]`, theta `5000000`, and max positions `262144`.
Existing data, deployment, optimizer, and checkpoint choices remain explicit.
Do not resume an old 1D optimizer state as the same experiment: the resume
contract now records the position convention and mRoPE settings. A deliberate
weights-only initialization is a separate training decision.

The capture payload stores position triplets as int64 `[1,S,3]`; the collator
produces `[B,S,3]`. Only the model boundary converts them to `[3,B,S]`. Missing
positions or a 1D draft config fail instead of silently training with 1D RoPE.
Context keys use the captured triplets, and candidate Q/K positions advance all
axes from the anchor, including slots beyond the returned training sequence.
Attention visibility and cache indices remain logical token indices.

Image geometry is computed from the target processor's merged grid. Explicit
image placement in message list content is preserved. Legacy top-level image
records without an explicit location still attach the image to the first user
turn. Multiple images and videos are rejected. The target capture service must
use the same processor settings and its existing multimodal capture patch with
`SGLANG_MM_AVOID_RETOKENIZE=1`; the producer's length check is not a substitute
for a real token/pixel identity audit.

The multimodal provider now honors `training.dflash_teacher_metrics` when
requesting target final hidden states. Text-only DFlash keeps its original 1D
path. DSpark training and the separate DFlash2-mRoPE experiment trees have not
been migrated by this change.

CPU regressions are in `tests/test_algorithms/test_dflash_vl_mrope.py`. They
cover Transformers coordinates and rotary parity, BF16 frequency preservation,
capture serialization, collation, strategy forwarding, a tiny model backward,
tail candidate continuation, image placement, and teacher capture selection.
No GPU performance, full target capture, or current serving-engine parity is
claimed by these unit tests.
