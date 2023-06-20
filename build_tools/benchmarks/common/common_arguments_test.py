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
        self.traced_tool_dir = self.build_dir / "traced_tool"
        self.traced_tool_dir.mkdir()
        self.trace_capture_tool = self.build_dir / "tracy_capture"
        # Create capture tool with executable file mode.
        self.trace_capture_tool.touch(mode=0o755)
        self.execution_config = self.build_dir / "execution_config.json"
        self.execution_config.touch()

    def tearDown(self):
        self._build_dir_manager.cleanup()

    def test_parser(self):
        common.common_arguments.Parser().parse_args(
            [
                f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
                f"--traced_benchmark_tool_dir={self.traced_tool_dir}",
                f"--trace_capture_tool={self.trace_capture_tool}",
                f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                f"--execution_benchmark_config={self.execution_config}",
                "--target_device=test",
            ]
        )

    def test_parser_check_normal_benchmark_tool(self):
        arg_parser = common.common_arguments.Parser()
        with self.assertRaises(SystemExit):
            arg_parser.parse_args(
                [
                    "--normal_benchmark_tool_dir=nonexistent",
                    f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                    f"--execution_benchmark_config={self.execution_config}",
                    "--target_device=test",
                ]
            )

    def test_parser_check_traced_benchmark_tool(self):
        arg_parser = common.common_arguments.Parser()
        with self.assertRaises(SystemExit):
            arg_parser.parse_args(
                [
                    "--traced_benchmark_tool_dir=nonexistent",
                    f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                    f"--execution_benchmark_config={self.execution_config}",
                    "--target_device=test",
                ]
            )

    def test_parser_check_trace_capture_tool(self):
        arg_parser = common.common_arguments.Parser()
        with self.assertRaises(SystemExit):
            arg_parser.parse_args(
                [
                    "--trace_capture_tool=nonexistent",
                    f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                    f"--execution_benchmark_config={self.execution_config}",
                    "--target_device=test",
                ]
            )


if __name__ == "__main__":
    unittest.main()
