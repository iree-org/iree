# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# cmake file utils for x86 benchmarks

# If NUM_OF_THREADS is empty, that means "run the local-sync" version.
function(iree_benchmark_suite_x86)
  cmake_parse_arguments(
    PARSE_ARGV 0
    X86_CONFIG
    # Options
    ""
    # Args with one value
    "TARGET_ARCHITECTURE;NUM_OF_THREADS;USE_EXPERIMENTAL_FLAGS;INPUT_TYPE"
    # Args with several values
    "MODULES"
  )

  if (NOT X86_CONFIG_TARGET_ARCHITECTURE)
    message(FATAL_ERROR "TARGET_ARCHITECTURE is mandatory")
  endif()
  string(TOLOWER ${X86_CONFIG_TARGET_ARCHITECTURE} TARGET_CPU)
  set(TARGET_ARCHITECTURE "CPU-x86_64-${X86_CONFIG_TARGET_ARCHITECTURE}")
  set(LINUX_X86_64_CPU_COMPILATION_FLAGS
    "--iree-input-type=${X86_CONFIG_INPUT_TYPE}"
    "--iree-llvm-target-cpu=${TARGET_CPU}"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
  )

  set(BENCHMARK_MODES "full-inference")
  if (X86_CONFIG_USE_EXPERIMENTAL_FLAGS)
    # Specialized benchmark configurations.
    set(BENCHMARK_MODES "${BENCHMARK_MODES},experimental-flags")
    list(APPEND LINUX_X86_64_CPU_COMPILATION_FLAGS
      "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
      "--iree-llvmcpu-enable-pad-consumer-fusion"
    )
  else()
    set(BENCHMARK_MODES "${BENCHMARK_MODES},default-flags")
  endif()

  if (NOT X86_CONFIG_NUM_OF_THREADS)
    set(CONFIG "iree-llvm-cpu-sync")
    set(DRIVER "local-sync")
  else()
    set(BENCHMARK_MODES "${X86_CONFIG_NUM_OF_THREADS}-thread,${BENCHMARK_MODES}")
    set(CONFIG "iree-llvm-cpu")
    set(DRIVER "local-task")
    set(RUNTIME_FLAGS "--task_topology_group_count=${X86_CONFIG_NUM_OF_THREADS}")
  endif()

  # CPU, LLVM, local-sync, x86_64, full-inference
  iree_benchmark_suite(
    GROUP_NAME
      "linux-x86_64"

    MODULES
     ${X86_CONFIG_MODULES}
    BENCHMARK_MODES
      "${BENCHMARK_MODES}"
    TARGET_BACKEND
      "llvm-cpu"
    TARGET_ARCHITECTURE
      "${TARGET_ARCHITECTURE}"
    COMPILATION_FLAGS
      ${LINUX_X86_64_CPU_COMPILATION_FLAGS}
    BENCHMARK_TOOL
      iree-benchmark-module
    CONFIG
      "${CONFIG}"
    DRIVER
      "${DRIVER}"
    RUNTIME_FLAGS
      "${RUNTIME_FLAGS}"
  )
endfunction(iree_benchmark_suite_x86)

