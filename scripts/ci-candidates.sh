#!/usr/bin/env bash
# Candidate artifacts remain private worker inputs; no native inference here.
set -euo pipefail
python_bin="${CFETCH_POLICY_PYTHON:-python3}"
export PYTHONPATH="${CFETCH_CANDIDATE_PYTHONPATH:-}${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m unittest discover -s experiments/embedding-candidates -v
"$python_bin" experiments/embedding-candidates/probe.py --help >/dev/null
if [[ -n "${CFETCH_CANDIDATE_ROOT:-}" ]]; then
  : "${CFETCH_CANDIDATE_REPORT:?new output file required}"
  "$python_bin" experiments/embedding-candidates/candidates.py "$CFETCH_CANDIDATE_ROOT" "$CFETCH_CANDIDATE_REPORT"
fi
