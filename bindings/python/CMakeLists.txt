# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NUMPY_DEPS "")

set(PYBIND_COPTS "-fexceptions")
set(PYBIND_EXTENSION_COPTS "-fvisibility=hidden")

# Namespace packages.
add_subdirectory(iree/runtime)
add_subdirectory(iree/jax)

if(IREE_BUILD_COMPILER AND IREE_BUILD_OLD_PYTHON_COMPILER_API)
add_subdirectory(iree/compiler)
add_subdirectory(iree/tools/core)
endif()

# Tests.
add_subdirectory(tests)
