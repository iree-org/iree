# Copyright 2022 The IREE Authors
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
import iree.runtime.version as v

assert hasattr(v, "PACKAGE_SUFFIX")
assert v.REVISIONS["IREE"]
assert v.VERSION
print("IREE version:", v.VERSION)

check_tool("iree-benchmark-module", ["--help"], "IREE: iree-benchmark-module")
check_tool("iree-benchmark-trace", ["--help"], "IREE: iree-benchmark-trace")
check_tool("iree-run-module", ["--help"], "IREE: iree-run-module")
check_tool("iree-run-trace", ["--help"], "IREE: iree-run-trace")
check_tool("iree-dump-module", ["--help"], "IREE: iree-dump-module")
check_tool("iree-dump-parameters", ["--help"], "IREE: iree-dump-parameters")
check_tool("iree-cpuinfo", [])

print("***** All done *****")
