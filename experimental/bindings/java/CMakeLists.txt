# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(com/google/iree/native)

if (${IREE_BUILD_TESTS})
  add_subdirectory(com/google/iree/tests)
endif()
