## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities that help generate artifacts."""

import re

# The characters unsafe for CMake and file systems. It's the negation of the
# allowed list, which is derived from the join of
# https://cmake.org/cmake/help/v3.27/policy/CMP0037.html and
# https://www.mtu.edu/umc/services/websites/writing/characters-avoid/.
UNSAFE_CHARACTERS = re.compile(r"[^0-9a-zA-Z\-_.]")


def get_safe_name(name: str) -> str:
    """Replace unsafe characters for CMake and file systems with `_`."""
    return UNSAFE_CHARACTERS.sub("_", name)
