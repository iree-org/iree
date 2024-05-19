#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest
import tempfile

from common import benchmark_config, benchmark_definition, common_arguments


class BenchmarkConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir_manager = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self._tmp_dir_manager.name).resolve()
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
        self._tmp_dir_manager.cleanup()

    def test_build_from_args(self):
        args = common_arguments.Parser().parse_args(
            [
                f"--tmp_dir={self.tmp_dir}",
                f"--benchmark_tool_dir={self.normal_tool_dir}",
                f"--driver_filter_regex=a",
                f"--model_name_regex=b",
                f"--mode_regex=c",
                f"--keep_going",
                f"--benchmark_min_time=10",
                f"--compatible_only",
                f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                f"--execution_benchmark_config={self.execution_config}",
                "--target_device=test",
                "--verify",
            ]
        )

        config = benchmark_config.BenchmarkConfig.build_from_args(
            args=args, git_commit_hash="abcd"
        )

        per_commit_tmp_dir = self.tmp_dir / "abcd"
        expected_config = benchmark_config.BenchmarkConfig(
            tmp_dir=per_commit_tmp_dir,
            root_benchmark_dir=benchmark_definition.ResourceLocation.build_local_path(
                self.e2e_test_artifacts_dir
            ),
            benchmark_results_dir=per_commit_tmp_dir / "benchmark-results",
            git_commit_hash="abcd",
            benchmark_tool_dir=self.normal_tool_dir,
            driver_filter="a",
            model_name_filter="b",
            mode_filter="c",
            keep_going=True,
            benchmark_min_time=10,
            use_compatible_filter=True,
            verify=True,
        )
        self.assertEqual(config, expected_config)

    def test_build_from_args_benchmark_only(self):
        args = common_arguments.Parser().parse_args(
            [
                f"--tmp_dir={self.tmp_dir}",
                f"--benchmark_tool_dir={self.normal_tool_dir}",
                f"--e2e_test_artifacts_dir={self.e2e_test_artifacts_dir}",
                f"--execution_benchmark_config={self.execution_config}",
                "--target_device=test",
            ]
        )

        config = benchmark_config.BenchmarkConfig.build_from_args(
            args=args, git_commit_hash="abcd"
        )

    def test_build_from_args_with_test_artifacts_dir_url(self):
        args = common_arguments.Parser().parse_args(
            [
                f"--tmp_dir={self.tmp_dir}",
                f"--benchmark_tool_dir={self.normal_tool_dir}",
                f"--e2e_test_artifacts_dir=https://example.com/testdata",
                f"--execution_benchmark_config={self.execution_config}",
                "--target_device=test",
            ]
        )

        config = benchmark_config.BenchmarkConfig.build_from_args(
            args=args, git_commit_hash="abcd"
        )

        self.assertEqual(
            config.root_benchmark_dir.get_url(), "https://example.com/testdata"
        )


if __name__ == "__main__":
    unittest.main()
