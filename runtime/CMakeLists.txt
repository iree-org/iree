# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(src)

if(IREE_BUILD_PYTHON_BINDINGS)
  # Copy Python packaging files to the build dir so that we can install from
  # there.
  configure_file(pyproject.toml pyproject.toml COPYONLY)
  configure_file(setup.py setup.py @ONLY)

  add_subdirectory(bindings/python)
endif()
