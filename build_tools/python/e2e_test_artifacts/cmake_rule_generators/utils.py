## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common helpers for cmake rule generators."""

import abc

# CMake variable name to store IREE package name.
PACKAGE_NAME_CMAKE_VARIABLE = "_PACKAGE_NAME"


def build_target_path(target_name: str):
  """Returns the full target path by combining the variable of package name and
  the target name.
  """
  return f"${{{PACKAGE_NAME_CMAKE_VARIABLE}}}_{target_name}"


class CMakeRule(abc.ABC):

  @abc.abstractmethod
  def get_rule(self) -> str:
    """Get cmake rule."""
    pass
