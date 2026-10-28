# Text training environment

Verified on 2026-10-28 with NVIDIA H100 80GB HBM3, driver 595.91.07, Linux x86_64, Python 3.11.16, Torch 2.13.0, Transformers 5.12.1, SGLang 0.5.18 and Triton 3.7.1. Source environment: specforge_main; 225 packages are locked.

## Install

~~~bash
git clone -b xiaojing/text-training https://github.com/zzz589/SpecForge.git
cd SpecForge
bash scripts/bootstrap_env.sh
source .venv/bin/activate
python scripts/verify_environment.py
~~~

For an offline cluster, download the locked wheels first and run:

~~~bash
WHEELHOUSE=/path/to/wheels bash scripts/bootstrap_env.sh
~~~

PIP_INDEX_URL and PIP_EXTRA_INDEX_URL are honored. The setup applies the SGLang v0.5.18 spec-capture patch and verifies its exact content.

## Container

~~~bash
docker build -f docker/Dockerfile.h100 -t specforge-text:h100 .
docker run --rm --gpus all --ipc=host --shm-size=32g -v /models:/models -v /data:/data -v /outputs:/outputs specforge-text:h100
~~~

The venv is the exact verified path. The pinned container is the portable secondary path; smoke-test it on each cluster.

Model weights, datasets, credentials, Mooncake services and outputs are external assets and must not be committed. See TRAINING_GUIDE.md for the complete workflow.
