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

if(${IREE_BUILD_COMPILER})
add_subdirectory(iree/compiler)
add_subdirectory(iree/tools/core)
endif()

if(${IREE_BUILD_XLA_COMPILER})
add_subdirectory(iree/jax)
endif()

# Tests.
add_subdirectory(tests)
