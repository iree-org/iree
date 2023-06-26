# Lint-as: python3
"""TensorFlow tools."""

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import os
import platform


def get_tool(exe_name: str) -> Optional[str]:
    if platform.system() == "Windows":
        exe_name = exe_name + ".exe"
    this_path = os.path.dirname(__file__)
    tool_path = os.path.join(this_path, exe_name)
    return tool_path
