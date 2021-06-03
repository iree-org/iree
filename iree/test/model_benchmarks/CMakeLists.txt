# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models                                                             #
#                                                                              #
# All the lists should have the same number of elements. Each list describes   #
# one aspect of the model. Elements at the same index are for the same model.  #
#                                                                              #
# Normally models to be benchmarked should be placed here becuase all          #
# benchmark cases will be enabled for them. There might exist cases where we   #
# cannot enable all the benchmark cases for one model; then they should be     #
# placed directly in the cmake function calls in the next section.             #
#                                                                              #
################################################################################

set(BENCHMARK_MODULE_NAMES
  "MobileNetV2"
  "MobileNetV3Small"
)

# Each element is a comma-separated list.
set(BENCHMARK_MODULE_TAGS
  "fp32,imagenet"
  "fp32,imagenet"
)

set(BENCHMARK_MODULE_SOURCES
  "TensorFlow"
  "TensorFlow"
)

set(BENCHMARK_MLIR_SOURCES
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV2-b0c5c584.tar.gz"
  "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3Small-b0c5c584.tar.gz"
)

set(BENCHMARK_ENTRY_FUNCTIONS
  "call"
  "call"
)

# Each element is a comma-separated list.
set(BENCHMARK_FUNCTION_INPUTS
  "1x224x224x3xf32"
  "1x224x224x3xf32"
)

################################################################################
#                                                                              #
# Benchmark cases                                                              #
#                                                                              #
################################################################################

# CPU, VMVX, 3-thread, little-core, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "3-thread,little-core,full-inference"
  TARGET_BACKEND
    "vmvx"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-flow-inline-constants-max-byte-length=2048"
  DRIVER
    "vmvx"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)

# CPU, Dylib-Sync, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "big-core,full-inference"
    "little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "1-thread,big-core,full-inference"
    "1-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, Dylib, 3-thread, big/little-core, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "3-thread,big-core,full-inference"
    "3-thread,little-core,full-inference"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-inline-constants-max-byte-length=2048"
    # TODO(GH-5857): Enable this after fixing segfault.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-llvm-loop-unrolling=true"
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=3"
)

# GPU, Vulkan, Adreno, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=adreno-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=2048"
    # TODO(GH-6086): Turn on the flag.
    # "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Adreno, kernel-execution
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Adreno"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=adreno-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=2048"
    # TODO(GH-6086): Turn on the flag.
    #"--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)

# GPU, Vulkan, Mali, full-inference
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "full-inference"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    # TODO(GH-6086): Turn on the flag.
    # "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
  DRIVER
    "vulkan"
)

# GPU, Vulkan, Mali, kernel-execution
iree_mlir_benchmark_suite(
  MODULE_NAMES
    ${BENCHMARK_MODULE_NAMES}
  MODULE_TAGS
    ${BENCHMARK_MODULE_TAGS}
  MODULE_SOURCES
    ${BENCHMARK_MODULE_SOURCES}
  MLIR_SOURCES
    ${BENCHMARK_MLIR_SOURCES}
  ENTRY_FUNCTIONS
    ${BENCHMARK_ENTRY_FUNCTIONS}
  FUNCTION_INPUTS
    ${BENCHMARK_FUNCTION_INPUTS}

  BENCHMARK_MODES
    "kernel-execution"
  TARGET_BACKEND
    "vulkan-spirv"
  TARGET_ARCHITECTURE
    "GPU-Mali-Valhall"
  TRANSLATION_FLAGS
    "--iree-input-type=mhlo"
    "--iree-vulkan-target-triple=valhall-unknown-android11"
    "--iree-flow-inline-constants-max-byte-length=16"
    # TODO(GH-6086): Turn on the flag.
    # "--iree-flow-dispatch-formation-enable-operand-fusion"
    "--iree-enable-fusion-with-reduction-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  DRIVER
    "vulkan"
  RUNTIME_FLAGS
    "--batch_size=32"
)
