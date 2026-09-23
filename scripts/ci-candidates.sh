#!/usr/bin/env bash
# Candidate artifacts remain private worker inputs; no native inference here.
set -euo pipefail
python_bin="${CFETCH_POLICY_PYTHON:-python3}"
export PYTHONPATH="${CFETCH_CANDIDATE_PYTHONPATH:-}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${CFETCH_MODEL_BUILD_LIBRARIES:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cxx_library=$("${CXX:-c++}" -print-file-name=libstdc++.so.6)
if [[ -f "$cxx_library" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$cxx_library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$python_bin" -m unittest discover -s experiments/embedding-candidates -v
"$python_bin" experiments/embedding-candidates/probe.py --help >/dev/null
if [[ -n "${CFETCH_CANDIDATE_ROOT:-}" ]]; then
  : "${CFETCH_CANDIDATE_REPORT:?new output file required}"
  "$python_bin" experiments/embedding-candidates/candidates.py "$CFETCH_CANDIDATE_ROOT" "$CFETCH_CANDIDATE_REPORT"
fi
