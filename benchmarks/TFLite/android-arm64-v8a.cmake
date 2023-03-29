# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
#                                                                              #
# Default benchmark configurations                                             #
#                                                                              #
# Each suite benchmarks a list of modules with configurations specifying a     #
# target architecture and runtime characteristics (e.g. threads/cores). These  #
# benchmarks only configure IREE compilation and runtime flags for the target  #
# architecture and do *not* include any non-default flags. No non-default      #
# flags should be added here.                                                  #
#                                                                              #
################################################################################

set(ANDROID_CPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvmcpu-target-triple=aarch64-none-linux-android29")

# CPU, LLVM, local-sync, big/little-core, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,default-flags"
    "little-core,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,default-flags"
    "1-thread,little-core,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# TODO(#7792): Re-enable these when we are able to run different benchmarks
# depending on use-case (presubmit, postsubmit, nightly, etc.)
# iree_benchmark_suite(
#   GROUP_NAME
#     "android-arm64-v8a"
#
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "2-thread,big-core,full-inference,default-flags"
#     "2-thread,little-core,full-inference,default-flags"
#   TARGET_BACKEND
#     "llvm-cpu"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   COMPILATION_FLAGS
#     ${ANDROID_CPU_COMPILATION_FLAGS}
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   CONFIG
#    "iree-llvm-cpu"
#   DRIVER
#     "local-task"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=2"
# )

# iree_benchmark_suite(
#   GROUP_NAME
#     "android-arm64-v8a"
#
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "3-thread,big-core,full-inference,default-flags"
#     "3-thread,little-core,full-inference,default-flags"
#   TARGET_BACKEND
#     "llvm-cpu"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   COMPILATION_FLAGS
#     ${ANDROID_CPU_COMPILATION_FLAGS}
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   CONFIG
#    "iree-llvm-cpu"
#   DRIVER
#     "local-task"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=3"
# )

iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,default-flags"
    "4-thread,little-core,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

################################################################################

################################################################################
#                                                                              #
# Specialized benchmark configurations                                         #
#                                                                              #
# Each suite benchmarks one or more module with configurations that can vary   #
# on model or architecture characteristics. These are intended for providing   #
# continuous benchmarks of experimental features that cannot be turned on by   #
# default yet. It is primarily intended for whoever is actively investigating  #
# optimizations for a feature exemplified in a specific model or architecture. #
# Due to our current benchmark setup, there can only be one experimental       #
# configuration per model and other benchmark mode.                            #
#                                                                              #
################################################################################

# CPU, LLVM, local-sync, big/little-core, full-inference
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. +f32mm).
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,experimental-flags"
    "little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# CPU, LLVM, local-sync, big/little-core, full-inference
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. +f32mm).
# TODO(#12788) For these benchmarks fusion results in stack allocations
# that are not found to be bounded which results in a compilation error.
# For now add a flag to ignore that error.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,experimental-flags"
    "little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)


# CPU, LLVM, local-sync, big/little-core, full-inference, +dotprod
# NOTE: +dotprod is only relevant to int8, not fp32.
# TODO: add a +i8mm variant, supported by new devices already. No rush: our i8mm
# kernel is currently naive, not ready for benchmarking.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILEBERT_INT8_MODULE}"

  BENCHMARK_MODES
    "big-core,full-inference,experimental-flags"
    "little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# TODO(#7792): Consider re-enabling little-core experimental-flags if we start
# optimizing for little cores or we can just run them occasionally

# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference.
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. f32mm).
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,experimental-flags"
    # "1-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference.
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. f32mm).
# TODO(#12788) For these benchmarks fusion results in stack allocations
# that are not found to be bounded which results in a compilation error.
# For now add a flag to ignore that error.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,experimental-flags"
    # "1-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)


# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference, +dotprod
# NOTE: +dotprod is only relevant to int8, not fp32.
# TODO: add a +i8mm variant, supported by new devices already. No rush: our i8mm
# kernel is currently naive, not ready for benchmarking.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILEBERT_INT8_MODULE}"

  BENCHMARK_MODES
    "1-thread,big-core,full-inference,experimental-flags"
    # "1-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# TODO(#7792): Re-enable these when we are able to run different benchmarks
# depending on use-case (presubmit, postsubmit, nightly, etc.)
# iree_benchmark_suite(
#  GROUP_NAME
#    "android-arm64-v8a"
#
#   MODULES
#     "${DEEPLABV3_FP32_MODULE}"
#     "${MOBILESSD_FP32_MODULE}"
#     "${POSENET_FP32_MODULE}"
#     "${MOBILEBERT_FP32_MODULE}"
#     "${MOBILENET_V2_MODULE}"
#     "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "2-thread,big-core,full-inference,experimental-flags"
#     "2-thread,little-core,full-inference,experimental-flags"
#   TARGET_BACKEND
#     "llvm-cpu"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   COMPILATION_FLAGS
#     ${ANDROID_CPU_COMPILATION_FLAGS}
#     "--iree-flow-enable-data-tiling"
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   CONFIG
#    "iree-llvm-cpu"
#   DRIVER
#     "local-task"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=2"
# )

# iree_benchmark_suite(
#  GROUP_NAME
#    "android-arm64-v8a"
#
#   MODULES
#   "${DEEPLABV3_FP32_MODULE}"
#   "${MOBILESSD_FP32_MODULE}"
#   "${POSENET_FP32_MODULE}"
#   "${MOBILEBERT_FP32_MODULE}"
#   "${MOBILENET_V2_MODULE}"
#   "${MOBILENET_V3SMALL_MODULE}"

#   BENCHMARK_MODES
#     "3-thread,big-core,full-inference,experimental-flags"
#     "3-thread,little-core,full-inference,experimental-flags"
#   TARGET_BACKEND
#     "llvm-cpu"
#   TARGET_ARCHITECTURE
#     "CPU-ARM64-v8A"
#   COMPILATION_FLAGS
#     ${ANDROID_CPU_COMPILATION_FLAGS}
#     "--iree-flow-enable-data-tiling"
#   BENCHMARK_TOOL
#     iree-benchmark-module
#   CONFIG
#    "iree-llvm-cpu"
#   DRIVER
#     "local-task"
#   RUNTIME_FLAGS
#     "--task_topology_group_count=3"
# )

# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference.
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. f32mm).
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
    # "4-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"

  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# CPU, LLVM, local-task, 1 through 4 threads, big/little-core, full-inference.
# NOTE: this is not enabling any SIMD extension beyond baseline Aarch64.
# At the moment we use that for fp32 models. We would change that when new
# devices support relevant fp32 SIMD extensions beyond that (e.g. f32mm).
# TODO(#12788) For these benchmarks fusion results in stack allocations
# that are not found to be bounded which results in a compilation error.
# For now add a flag to ignore that error.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
    # "4-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false"

  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# CPU, LLVM, local-sync, big/little-core, full-inference, +dotprod
# NOTE: +dotprod is only relevant to int8, not fp32.
# TODO: add a +i8mm variant, supported by new devices already. No rush: our i8mm
# kernel is currently naive, not ready for benchmarking.
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILEBERT_INT8_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
    # "4-thread,little-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    ${ANDROID_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-data-tiling"
    "--iree-llvmcpu-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"

  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# CPU, VMVX, 4-thread, big-core, full-inference
# VMVX is slow and we're not optimizing perf yet. Leaving in a single max-thread
# benchmark because it's useful to keep an eye on and helps disambiguate where a
# performance change may be coming from (e.g. if it's in vmvx as well, it's
# probably not a codegen issue).
iree_benchmark_suite(
  GROUP_NAME
    "android-arm64-v8a"

  MODULES
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,big-core,full-inference,experimental-flags"
  TARGET_BACKEND
    "vmvx"
  TARGET_ARCHITECTURE
    "CPU-ARM64-v8A"
  COMPILATION_FLAGS
    "--iree-input-type=tosa"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-vmvx"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

################################################################################
