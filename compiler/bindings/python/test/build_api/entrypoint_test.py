# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from iree.build.__main__ import main


class BuildEntrypointTest(unittest.TestCase):
    def testConsoleScriptMainIsCallable(self):
        # The `iree-build` console script resolves to this symbol.
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
