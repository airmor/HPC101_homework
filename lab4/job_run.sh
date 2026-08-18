#!/bin/bash
set -euo pipefail
LAB=/home/h3250105245/HPC101_homework/lab4
cd "$LAB"
export AMSS_BUILD_DIR="$LAB/build"
export AMSS_OUTPUT_ROOT="$LAB"
export AMSS_CACHE_DIR="$LAB/twopuncture_cache"
# OpenMPI 5/PRRTE fails to read the cgroup cpuset (30 cores in the container,
# but default slots < 30), so -n 30 reports "not enough slots". --oversubscribe
# lets it launch the requested ranks. run.sh honors AMSS_MPIEXEC.
export AMSS_MPIEXEC="mpiexec --allow-run-as-root --oversubscribe"
echo "JOB_RUN pwd=$(pwd) BUILD=$AMSS_BUILD_DIR MPIEXEC=$AMSS_MPIEXEC"
exec ./run.sh "$@"
