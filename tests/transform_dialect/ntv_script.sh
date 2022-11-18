#!/bin/sh


iree-presubmits() {
  THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
  # Set IREE_SOURCE_DIR for us if not already set
  IREE_SOURCE_DIR=${IREE_SOURCE_DIR:=${THIS_SCRIPT_DIR}/../..}
  python ${IREE_SOURCE_DIR}/build_tools/bazel_to_cmake/bazel_to_cmake.py
  (cd ${IREE_SOURCE_DIR} && ./build_tools/scripts/run_buildifier.sh)
}

iree-transform-get-args(){
  usage() { 
    echo 1>&2 'Usage: iree-transform-xxx <mlir-input-file> -b <backend> [-c <codegen-spec-file>] [-d <dispatch-spec-file>] [-s <jitter-strategy-string>] [-- extra arguments]'
  }
  
  MLIR_FILE=$1
  local OPTIND o BACKEND CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE
  OPTIND=2
  while getopts ":c:d:b:" o; do
    case "${o}" in
      b)
        BACKEND=${OPTARG}
        ;;
      c)
        CODEGEN_SPEC_FILE=${OPTARG}
        ;;
      d)
        DISPATCH_SPEC_FILE=${OPTARG}
        ;;
      *)
        opts+=("-${opt}"); [[ -n "$OPTARG" ]] && opts+=("$OPTARG")
        ;;
    esac
  done
  shift $(expr $OPTIND - 1 )

  if [ -z "${BACKEND}" ] || [ -z "${MLIR_FILE}" ] ; then
    usage
    return 1
  fi
 
  MAYBE_EMPTY_CODEGEN_SPEC_FILE=${CODEGEN_SPEC_FILE:=/dev/null}
  MAYBE_EMPTY_DISPATCH_SPEC_FILE=${DISPATCH_SPEC_FILE:=/dev/null}
  # For debugging purposes
  # echo BACKEND=~~~${BACKEND}~~~MLIR_FILE=~~~${MLIR_FILE}~~~\
  # CODEGEN_SPEC_FILE=~~~${CODEGEN_SPEC_FILE}~~~\
  # MAYBE_EMPTY_DISPATCH_SPEC_FILE=~~~${MAYBE_EMPTY_DISPATCH_SPEC_FILE}~~~\
  # REST=~~~${@}~~~
  echo ${BACKEND} ${MLIR_FILE} ${MAYBE_EMPTY_CODEGEN_SPEC_FILE} ${MAYBE_EMPTY_DISPATCH_SPEC_FILE} ${@}
}

# iree-transform-opt-dispatch-only ./tests/transform_dialect/cpu/matmul.mlir \
#   -b llvm-cpu \
#   [ -c ./tests/transform_dialect/cpu/matmul_codegen_spec.mlir ] \
#   [ -d ./tests/transform_dialect/cpu/matmul_dispatch_spec.mlir ] \
#   [ -- extra_stuff ]
iree-transform-opt-dispatch-only() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})
  
  if test ${DISPATCH_SPEC_FILE} == /dev/null; then
    DISPATCH_FLAG="--iree-flow-enable-aggressive-fusion"
  else
    DISPATCH_FLAG="--iree-flow-dispatch-use-transform-dialect=${DISPATCH_SPEC_FILE}"
  fi

  iree-opt ${MLIR_FILE} \
    --iree-abi-transformation-pipeline \
    --iree-flow-transformation-pipeline \
    ${DISPATCH_FLAG} \
    ${EXTRA_ARGS}
}

# iree-transform-opt ./tests/transform_dialect/cpu/matmul.mlir \
#   -b llvm-cpu \
#   [ -c ./tests/transform_dialect/cpu/matmul_codegen_spec.mlir ] \
#   [ -d ./tests/transform_dialect/cpu/matmul_dispatch_spec.mlir ] \
#   [ -s cpu-matmul-strategy ]
#   [ extra_stuff ]
iree-transform-opt() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})
  
  if test ${BACKEND} == cuda; then
    # Note the discrepancy: iree-llvmgpu-lower-executable-target
    CODEGEN_FLAG="--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))"
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmgpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmgpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
  elif test ${BACKEND} == llvm-cpu; then
    # Note the discrepancy: iree-llvmcpu-lower-executable-target
    CODEGEN_FLAG="--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))"
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmcpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmcpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
  else
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi


  echo "iree-transform-opt-dispatch-only ${MLIR_FILE} -b ${BACKEND} -c ${CODEGEN_SPEC_FILE} -d ${DISPATCH_SPEC_FILE} | \ "
  echo "iree-opt --iree-hal-target-backends=${BACKEND} --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | \ "
  echo "iree-opt ${CODEGEN_FLAG} ${EXTRA_ARGS}"

  iree-transform-opt-dispatch-only ${MLIR_FILE} -b ${BACKEND} -c ${CODEGEN_SPEC_FILE} -d ${DISPATCH_SPEC_FILE} | \
  iree-opt --iree-hal-target-backends=${BACKEND} --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | \
  iree-opt ${CODEGEN_FLAG} ${EXTRA_ARGS}
}

# Pipe this through iree-run-module, e.g.:
#
#   iree-transform-compile -b llvm-cpu -i i.mlir -c c.mlir -d d.mlir -- \
#       --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host | \
#     iree-run-module --entry_function=max_sub_exp --device=local-task \
#       --function_input="32x1024xf32=1" --function_input="32xf32=-1066"
#
#   iree-transform-compile -b llvm-cpu -i i.mlir -c c.mlir -d d.mlir -- \
#       --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host --iree-hal-benchmark-dispatch-repeat-count=100 | \
#     iree-benchmark-module --device=local-task --task_topology_group_count=0 \
#       --batch_size=100 --entry_function=reduce \
#       --function_input="32x1024xf32=1" --function_input="32xf32=-1066"
#
iree-transform-compile() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})

  if test ${BACKEND} == "cuda"; then
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG="--iree-codegen-llvmgpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG="--iree-codegen-llvmgpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
  elif test ${BACKEND} == "llvm-cpu"; then
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG="--iree-codegen-llvmcpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG="--iree-codegen-llvmcpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
  else
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi
  if test ${DISPATCH_SPEC_FILE} == /dev/null; then
    DISPATCH_FLAG="--iree-flow-enable-aggressive-fusion"
  else
    DISPATCH_FLAG="--iree-flow-dispatch-use-transform-dialect=${DISPATCH_SPEC_FILE}"
  fi

  # WARNING!!!!!!!!!!!!!!!
  # Do not let this uncommented without redirecting to non-stdout otherwise 
  # iree-run-module will choke with an error resembing:
  # ```
  #   Reading module contents from stdin...
  #   iree/runtime/src/iree/vm/bytecode_module.c:133: INVALID_ARGUMENT; 
  #   FlatBuffer length prefix out of bounds (prefix is 1313756498 but only 12710 available)
  #
  # echo "RUNNING THE FOLLOWING COMMAND:"
  # echo """iree-compile ${MLIR_FILE} --iree-hal-target-backends=${BACKEND} \
  #   ${DISPATCH_FLAG} \
  #   ${CODEGEN_FLAG} \
  #   ${EXTRA_ARGS}"""

  iree-compile ${MLIR_FILE} --iree-hal-target-backends=${BACKEND} \
    ${DISPATCH_FLAG} \
    ${CODEGEN_FLAG} \
    ${EXTRA_ARGS}
}
