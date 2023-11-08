# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys

# Note that dump-module is only in the default libs.
from iree import _runtime_libs


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    exe = os.path.join(_runtime_libs.__path__[0], "iree-dump-parameters")
    return subprocess.call(args=[exe] + args)


if __name__ == "__main__":
    sys.exit(main())
