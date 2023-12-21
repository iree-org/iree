# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs on a pip installed runtime package and verifies it is setup properly."""

from typing import Optional

import subprocess
from typing import List


# Check tools.
def check_tool(tool_name: str, args: List[str], find_line: Optional[str] = None):
    print(f"**** Checking tool {tool_name} with args {args}")
    output = subprocess.check_output([tool_name] + args).decode()
    if find_line is not None:
        output_lines = output.splitlines()
        for line in output_lines:
            if find_line in line:
                print(f"Found output: {line.strip()}")
                return
        raise ValueError(
            f"Did not find banner '{find_line}' for {tool_name}:\n{output}"
        )


# Verify version.
import iree.compiler.version as v

assert hasattr(v, "PACKAGE_SUFFIX")
assert v.REVISIONS["IREE"]
assert v.VERSION
print("IREE version:", v.VERSION)

check_tool("iree-compile", ["--help"], "IREE compilation driver")
check_tool("iree-ir-tool", ["--help"], "IREE IR Tool")

# ONNX dependent.
onnx_available = False
try:
    import onnx

    onnx_available = True
except ModuleNotFoundError:
    print("Not checking iree-import-onnx: onnx pip package not found")
if onnx_available:
    check_tool("iree-import-onnx", ["--help"], "IREE ONNX import tool")

print("***** All done *****")
