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

if(IREE_BUILD_LEGACY_JAX)
  message(STATUS "Building legacy JAX API")
  add_subdirectory(iree/jax)
endif()
