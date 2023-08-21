# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(src)

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(bindings/python)
endif()
