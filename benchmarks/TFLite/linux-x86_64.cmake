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

set(LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvmcpu-target-cpu=cascadelake"
  "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu"
)

# CPU, LLVM, local-sync, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V1_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"
    "${MOBILENET_V2_INT8_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# CPU, LLVM, local-task, 1 thread, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "1-thread,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, LLVM, local-task, 4 threads, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "4-thread,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# CPU, LLVM, local-task, 8 threads, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "8-thread,full-inference,default-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=8"
)

################################################################################
#                                                                              #
# Specialized benchmark configurations                                         #
#                                                                              #
################################################################################

# CPU, LLVM, local-sync, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu-sync"
  DRIVER
    "local-sync"
)

# CPU, LLVM, local-task, 1 thread, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "1-thread,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
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

# CPU, LLVM, local-task, 4 threads, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "4-thread,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
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

# CPU, LLVM, local-task, 8 threads, x86_64, full-inference
iree_benchmark_suite(
  GROUP_NAME
    "linux-x86_64"

  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"
    "${MOBILEBERT_INT8_MODULE}"
    "${PERSON_DETECT_INT8_MODULE}"
    "${EFFICIENTNET_INT8_MODULE}"

  BENCHMARK_MODES
    "8-thread,full-inference,experimental-flags"
  TARGET_BACKEND
    "llvm-cpu"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  COMPILATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_COMPILATION_FLAGS}
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  BENCHMARK_TOOL
    iree-benchmark-module
  CONFIG
    "iree-llvm-cpu"
  DRIVER
    "local-task"
  RUNTIME_FLAGS
    "--task_topology_group_count=8"
)
