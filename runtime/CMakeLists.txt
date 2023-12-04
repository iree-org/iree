# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Must include runtime plugins before processing the runtime sources so that
# the static link list can be set.
iree_include_cmake_plugin_dirs(
  LOG_LABEL
    runtime
  BINARY_DIR
    "${IREE_BINARY_DIR}/runtime/plugins"
  PLUGIN_CMAKE_FILE
    "iree_runtime_plugin.cmake"
)

add_subdirectory(src)

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(bindings/python)
endif()
