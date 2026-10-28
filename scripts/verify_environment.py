#!/usr/bin/env python3
from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import subprocess
import sys
EXPECTED={"torch":"2.13.0","transformers":"5.12.1","sglang":"0.5.18","triton":"3.7.1","liger-kernel":"0.8.4","yunchang":"0.6.4"}
errors=[]
if sys.version_info[:2]!=(3,11): errors.append(f"expected Python 3.11, found {sys.version.split()[0]}")
for name,want in EXPECTED.items():
    try: got=md.version(name)
    except md.PackageNotFoundError: errors.append(f"missing {name}=={want}"); continue
    print(f"{name}: {got}")
    if got!=want: errors.append(f"{name}: expected {want}, found {got}")
try:
    import torch
    print(f"torch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available() and os.getenv("ALLOW_NO_CUDA")!="1": errors.append("CUDA unavailable")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()): print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as exc: errors.append(f"torch check failed: {exc}")
try:
    print(subprocess.check_output(["nvidia-smi","--query-gpu=driver_version,name","--format=csv,noheader"],text=True).strip())
except Exception as exc:
    if os.getenv("ALLOW_NO_CUDA")!="1": errors.append(f"nvidia-smi failed: {exc}")
try:
    import sglang
    root=Path(__file__).resolve().parents[1]
    patch=root/"patches"/"sglang"/"v0.5.18"/"spec-capture.patch"
    marker=Path(sglang.__file__).resolve().parent/".spec_capture_patch.applied"
    if not marker.is_file(): errors.append("SGLang spec-capture patch marker missing")
    elif marker.read_bytes()!=patch.read_bytes(): errors.append("installed SGLang patch differs from branch")
    else: print("SGLang spec-capture patch: verified")
except Exception as exc: errors.append(f"SGLang patch check failed: {exc}")
if errors:
    print("\nFAILED",file=sys.stderr)
    for error in errors: print(f"- {error}",file=sys.stderr)
    raise SystemExit(1)
print("\nEnvironment verification PASSED")
