# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path

sys.dont_write_bytecode = True

from iree import _runtime_libs


def main(arguments=None):
    if arguments is None:
        arguments = sys.argv[1:]
    profile_package_path = (
        Path(_runtime_libs.__path__[0]) / "share" / "iree" / "profile"
    )
    sys.path.insert(0, str(profile_package_path))
    from render.cli import main as render_main

    return render_main(arguments)


if __name__ == "__main__":
    sys.exit(main())
