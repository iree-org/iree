#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import tempfile
import unittest

import common.common_arguments


class CommonArgumentsTest(unittest.TestCase):
    def setUp(self):
        self._build_dir_manager = tempfile.TemporaryDirectory()
        self.build_dir = pathlib.Path(self._build_dir_manager.name).resolve()
        self.e2e_test_artifacts_dir = self.build_dir / "e2e_test_artifacts"
        self.e2e_test_artifacts_dir.mkdir()
        self.normal_tool_dir = self.build_dir / "normal_tool"
        self.normal_tool_dir.mkdir()
        self.execution_config = self.build_dir / "execution_config.json"
        self.execution_config.touch()

    def tearDown(self):
        self._build_dir_manager.cleanup()

    def test_parser(self):
        common.common_arguments.Parser().parse_args(
            [
                f"--benchmark_tool_dir={self.normal_tool_dir}",
                f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                f"--execution_benchmark_config={self.execution_config}",
                "--target_device=test",
            ]
        )

    def test_parser_check_benchmark_tool(self):
        arg_parser = common.common_arguments.Parser()
        with self.assertRaises(SystemExit):
            arg_parser.parse_args(
                [
                    "--benchmark_tool_dir=nonexistent",
                    f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                    f"--execution_benchmark_config={self.execution_config}",
                    "--target_device=test",
                ]
            )

if __name__ == "__main__":
    unittest.main()
