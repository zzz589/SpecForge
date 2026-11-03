# Qwen3-VL-4B DFlash2 on the PR785 multimodal path

This recipe composes two paths that already exist in the source tree:

1. PR785's `dflash` multimodal streaming provider preprocesses image/text
   records, sends images to a patched SGLang capture server, and transports the
   target's fused language hidden states to the trainer.
2. `DFlash2DraftModel` uses the normal `dflash` strategy and adds grouped local
   convolutions plus the candidate selector and its auxiliary loss.

No separate VLM DFlash2 model class is needed.  Pixels stay in the target
server; the DFlash2 draft sees the five captured target hidden-state layers.
Like the rest of PR785, draft positions use plain one-dimensional RoPE.

## Files

- Draft: `configs/qwen3-vl-4b-dflash2-b16.json`
- Run: `examples/configs/online/disaggregated/external/qwen3-vl-4b-dflash2-b16.yaml`
- Composition test: `tests/test_algorithms/test_dflash2_multimodal.py`

The five capture layers `[3, 10, 18, 25, 32]`, Qwen3-VL dimensions and special
tokens come from the existing Qwen3-VL DFlash work and the local target model.
The DFlash2 convolution/selector settings (`kernel=2`, `group=16`,
`rank=256`, `top_k=16`) follow the upstream DFlash2 configurations.  The run's
two-rank SDPA topology, 41,561-example mixed LLaVA corpus and
`producer_lease: 1` come from the completed PR785 Qwen3-VL DSpark run.

## Validation sequence

From the repository root, using the dedicated environment:

```bash
python -m pytest -q \
  tests/test_algorithms/test_dflash2_multimodal.py \
  tests/test_algorithms/test_dflash_multimodal.py \
  tests/test_modeling/test_dflash2.py

specforge train -c \
  examples/configs/online/disaggregated/external/qwen3-vl-4b-dflash2-b16.yaml \
  --plan
```

The dedicated environment is
`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/envs/specforge785-dflash2-vl`.
Its CUDA toolkit comes from the installed CUDA 13 wheel.  Background capture
services must receive the environment explicitly (an SSH/tmux session on hn47
does not inherit it):

```bash
ENV_ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/envs/specforge785-dflash2-vl
CUDA_ROOT="$ENV_ROOT/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$ENV_ROOT/bin:$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

The environment contains `lib64 -> lib` and
`lib/libcudart.so -> libcudart.so.13` compatibility links, needed by SGLang
0.5.18's JIT linker.  The capture server also needs
`SGLANG_MM_AVOID_RETOKENIZE=1`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`, the Mooncake
endpoint variables, and capture layers `3 10 18 25 32`.

For a live smoke test, start the same patched SGLang 0.5.18 Qwen3-VL capture
server and Mooncake services used by the PR785 DSpark run, then override
`training.max_steps` to a small value before launching producer and consumer.
Do not reuse a control or consumer-state directory from an earlier run.

## Live smoke result (2026-11-03, hn47)

The composition was exercised end to end with the patched SGLang 0.5.18 target
on GPU 5 and one DFlash2 trainer rank on GPU 3:

- 32 image/text records were prepared with no skips;
- two optimizer steps completed with finite total losses `12.2963` and
  `23.7950` (CE `12.2963` and `22.3983`);
- DFlash2 selector metrics were emitted and selector loss moved from `0.0` to
  `8.4555` after the first update;
- a step-2 checkpoint was written, including rank training state;
- the producer published 96 references (32 records times 3 prompt epochs), and
  the consumer acknowledged the four samples needed for two accumulated steps;
- `consumer_done` was written and no failure marker appeared.

Artifacts and logs are under
`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments/qwen3-vl-4b-dflash2-b16-pr785-smoke`.

## Intentional limitations

- One image per sample, matching PR785 v1.
- Online external-server capture only; offline multimodal capture is absent.
- `dflash_teacher_metrics: false`, because the PR785 multimodal provider does
  not transport the final target hidden state.  This does not change the
  training objective or selector loss.
- No DFlash1 warm start.  The current strict loader rejects missing DFlash2
  convolution and selector tensors; the new modules instead use their stable
  identity/no-op initialization.
