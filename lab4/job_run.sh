#!/bin/bash
# job_run.sh — robust submit wrapper.
# hpc submit's default job cwd is not always the submit cwd; this wrapper
# explicitly cds into the lab dir (absolute path, shared across devpod +
# compute container via NFS home) before exec'ing run.sh, so run.sh's
# ROOT_DIR="$(pwd)" resolves correctly regardless of the container's initial
# cwd.
set -euo pipefail
cd /home/h3250105245/HPC101_homework/lab4
exec ./run.sh "$@"
