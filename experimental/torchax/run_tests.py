# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs each test module in its own process, using the installed package."""

import importlib
import pathlib
import subprocess
import sys


def main():
    # The suite must exercise both export and execution, not silently skip them.
    for name in (
        "iree.compiler",
        "iree.runtime",
        "iree_torchax",
        "jax",
        "torch",
        "torchax",
    ):
        importlib.import_module(name)

    failed = []
    for test in sorted(pathlib.Path(__file__).with_name("tests").glob("*_test.py")):
        print(f"Running {test.name}", flush=True)
        if subprocess.run([sys.executable, str(test)], check=False).returncode:
            failed.append(test.name)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
