# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cts_target(
  DRIVER_NAME
    cuda
  SRCS
    "driver_registration.cc"
  DEPS
    iree::hal::cuda::registration
  EXCLUDED_TESTS
    # Semaphores are not implemented in the CUDA backend yet.
    "semaphore_submission"
    "semaphore"
)
