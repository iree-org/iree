# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Always build the C bindings, since the API is available apart from
# actually building the compiler.
add_subdirectory(bindings/c)

# The compiler implementation is gated on the global setting.
if(IREE_BUILD_COMPILER)
  # Must configure plugins before processing the compiler sources so that
  # the static link list can be set.
  iree_compiler_configure_plugins()
  add_subdirectory(src)

  # Copy Python packaging files to the build dir so that we can install from
  # there.
  if(IREE_BUILD_PYTHON_BINDINGS)
    configure_file(pyproject.toml pyproject.toml COPYONLY)
    configure_file(setup.py setup.py @ONLY)
    add_subdirectory(bindings/python)
  endif()
endif()
