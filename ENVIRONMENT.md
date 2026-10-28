# Vision-language training environment

Verified on 2026-10-28 with NVIDIA H100 80GB HBM3, driver 595.91.07, Linux x86_64, Python 3.11.16, Torch 2.13.0, Transformers 5.12.1, SGLang 0.5.18 and Triton 3.7.1. Source environment: specforge785; 226 packages are locked.

The SGLang v0.5.18 spec-capture patch is mandatory for multimodal capture and DSpark target_last_hidden_states.

## Install

~~~bash
git clone -b xiaojing/vl-training https://github.com/zzz589/SpecForge.git
cd SpecForge
bash scripts/bootstrap_env.sh
source .venv/bin/activate
python scripts/verify_environment.py
~~~

For an offline cluster:

~~~bash
WHEELHOUSE=/path/to/wheels bash scripts/bootstrap_env.sh
~~~

PIP_INDEX_URL and PIP_EXTRA_INDEX_URL are honored.

## Container

~~~bash
docker build -f docker/Dockerfile.h100 -t specforge-vl:h100 .
docker run --rm --gpus all --ipc=host --shm-size=32g -v /models:/models -v /data:/data -v /outputs:/outputs specforge-vl:h100
~~~

The venv is the exact verified route. The pinned container is the portable secondary route and needs a smoke test on each cluster.

VL training also requires model weights, image files, regenerated JSONL, a patched capture server and reachable Mooncake metadata/master services. See TRAINING_GUIDE.md.
