# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

configure_file(pyproject.toml pyproject.toml COPYONLY)
configure_file(setup.py setup.py @ONLY)

iree_py_library(
  NAME
    distributed
  SRCS
    "iree/experimental/runtime/distributed/__init__.py"
    "iree/experimental/runtime/distributed/distributed.py"
    "iree/experimental/runtime/distributed/run_rank.py"
    "iree/experimental/runtime/distributed/sharding_pass_validation.py"
    "iree/experimental/runtime/distributed/utils.py"
    ${_PYTHON_EXTRA_SRCS}
)

set(_INSTALL_DIR "python_packages/iree_distributed")
set(_INSTALL_COMPONENT "IreePythonPackage-distributed")

# Install iree/experimental/distributed/*.py files verbatim into the tree.
# We do this at the package level so as to avoid any loose files
# from outside (i.e. tests, etc).
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iree/experimental/runtime/distributed/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree/experimental/runtime/distributed/"
  FILES_MATCHING PATTERN "*.py"
)
