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
include(../linux-x86_64-common.cmake)

# Produce all the configurations we care about.
foreach(TARGET ${IREE_BUILD_BENCHMARKS_FOR_X86_TARGETS})
  foreach(USE_EXPERIMENTAL_FLAGS TRUE FALSE)
    # local-sync run ("") then multi-threaded runs.
    foreach(NUM_OF_THREADS "" 1 4 8)
      iree_benchmark_suite_x86(
        TARGET_ARCHITECTURE "${TARGET}"
        NUM_OF_THREADS "${NUM_OF_THREADS}"
        USE_EXPERIMENTAL_FLAGS ${USE_EXPERIMENTAL_FLAGS}
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
        INPUT_TYPE tosa
      )
    endforeach()
  endforeach()
endforeach()
