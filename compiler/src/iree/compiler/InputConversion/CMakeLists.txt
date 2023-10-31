# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(Common)

if(IREE_INPUT_STABLEHLO)
  add_subdirectory(StableHLO)
endif()
if(IREE_INPUT_TOSA)
  add_subdirectory(TOSA)
endif()
