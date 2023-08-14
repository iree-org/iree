# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys


def main(args=None):
    try:
        from iree import _runtime_libs_tracy
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This command requires that a tracy runtime is available "
            "but it was not built for this platform."
        ) from e
    if args is None:
        args = sys.argv[1:]
    exe = os.path.join(_runtime_libs_tracy.__path__[0], "iree-tracy-capture")
    return subprocess.call(args=[exe] + args)


if __name__ == "__main__":
    sys.exit(main())
