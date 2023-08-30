#!/bin/bash

set -xeuo pipefail

IREE_OPT="$(which ${IREE_OPT:-iree-opt})"
IREE_COMPILE="$(which ${IREE_COMPILER:-iree-compile})"
IREE_BENCHMARK_MODULE="$(which ${IREE_BENCHMARK_MODULE:-iree-benchmark-module})"
IREE_TRACY="$(which ${IREE_TRACY:-iree-tracy-capture})"
TRACE_MODE="${TRACE_MODE:-0}"
THREADS="${THREADS:-1}"
PREFIX="${PREFIX:-}"
MODEL_DIR="${MODEL_DIR:-.}"
COMP_FLAGS="${COMP_FLAGS:-}"

# for MODEL_PATH in $(ls "${MODEL_DIR}/"*.mlirbc); do
for MODEL_PATH in $(ls "${MODEL_DIR}"/BertLargeTF_Batch32.mlirbc); do
  MODEL_FILE="$(basename "${MODEL_PATH}")"
  echo ">>>> ${MODEL_FILE} <<<<"

  "${IREE_COMPILE}" \
    "${MODEL_PATH}" \
    -o "${PREFIX}${MODEL_FILE}.linalg.mlir" \
    --iree-hal-target-backends=llvm-cpu \
    --iree-input-type=auto \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu=cascadelake \
    --iree-flow-enable-data-tiling \
    --iree-llvmcpu-enable-microkernels \
    --compile-to="preprocessing"

  "${IREE_OPT}" --mlir-print-debuginfo "${PREFIX}${MODEL_FILE}.linalg.mlir" > "${PREFIX}${MODEL_FILE}.debug.mlir"

  "${IREE_COMPILE}" \
    "${PREFIX}${MODEL_FILE}.debug.mlir" \
    -o "${PREFIX}${MODEL_FILE}.vmfb" \
    ${COMP_FLAGS} \
    --iree-hal-target-backends=llvm-cpu \
    --iree-input-type=auto \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu=cascadelake \
    --iree-flow-enable-data-tiling \
    --iree-llvmcpu-enable-microkernels \
    --mlir-print-ir-after=iree-flow-outline-dispatch-regions \
    --mlir-elide-elementsattrs-if-larger=4 2> "${PREFIX}${MODEL_FILE}.dump"

  if (( THREADS == 1 )); then
    declare -a THREAD_ARGS=(
      "--device=local-sync"
    )
  else
    declare -a THREAD_ARGS=(
      "--device=local-task"
      "--task_topology_max_group_count=${THREADS}"
    )
  fi

  RUN_ARGS=($(cat "${MODEL_PATH}.run_flag"))

  if (( TRACE_MODE == 1 )); then
    "${IREE_TRACY}" -f -o "${PREFIX}${MODEL_FILE}".tracy >/dev/null &
    REPETITIONS=1
  else
    REPETITIONS=5
  fi

  TRACY_NO_EXIT="${TRACE_MODE}" numactl --cpubind=0 --membind=0 -- \
    "${IREE_BENCHMARK_MODULE}" \
    --device_allocator=caching \
    --benchmark_repetitions="${REPETITIONS}" \
    --module=${PREFIX}${MODEL_FILE}.vmfb \
    "${THREAD_ARGS[@]}" \
    "${RUN_ARGS[@]}"

  wait
done
