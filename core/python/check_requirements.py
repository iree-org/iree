# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks that the dependencies in pyproject.toml are installed.

CMake registers the tests of the package only if this succeeds. It must not
import iree.core: a broken iree.core has to fail its tests, not hide them.
"""

import importlib.metadata
import pathlib
import re
import sys


def main() -> int:
    if sys.version_info < (3, 11):
        print("iree.core needs Python 3.11 or newer", file=sys.stderr)
        return 1
    import tomllib

    pyproject = pathlib.Path(__file__).parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]
    missing = []
    for requirement in dependencies:
        name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0]
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        print(
            f"Python packages that {pyproject} lists are not installed: "
            + ", ".join(missing)
            + " (install requirements.txt next to it)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
