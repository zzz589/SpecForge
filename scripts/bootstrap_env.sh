#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
WHEELHOUSE="${WHEELHOUSE:-}"
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python 3.11 is required" >&2; exit 1; }
"${PYTHON_BIN}" -m venv "${ENV_DIR}"
PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade "pip>=24,<27" "setuptools==84.0.0" wheel
PIP_ARGS=()
[[ -n "${WHEELHOUSE}" ]] && PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
"${PYTHON}" -m pip install "${PIP_ARGS[@]}" -r "${ROOT_DIR}/requirements-lock.txt"
"${PYTHON}" -m pip install --no-deps -e "${ROOT_DIR}"
PATH="${ENV_DIR}/bin:${PATH}" "${ROOT_DIR}/scripts/apply_sglang_spec_capture_patch.sh" --target v0.5.18
echo "Environment ready: ${ENV_DIR}"
echo "source ${ENV_DIR}/bin/activate"
echo "python scripts/verify_environment.py"
