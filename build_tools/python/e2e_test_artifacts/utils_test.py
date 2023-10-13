## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_test_artifacts import utils


class UtilsTest(unittest.TestCase):
    def test_get_safe_name_with_disallowed_characters(self):
        sanitized_name = utils.get_safe_name("test(abc) [x,y,z]")

        self.assertEqual(sanitized_name, "test_abc___x_y_z_")

    def test_get_safe_name_with_all_allowed_characters(self):
        safe_name = "123_AB-C.test"

        sanitized_name = utils.get_safe_name(safe_name)

        self.assertEqual(sanitized_name, safe_name)


if __name__ == "__main__":
    unittest.main()
