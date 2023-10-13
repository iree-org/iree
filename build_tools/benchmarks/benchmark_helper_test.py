#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import unittest
import benchmark_helper
import tempfile
import pathlib


class BenchmarkHelperTest(unittest.TestCase):
    def test_merge_results_simple(self):
        first = benchmark_helper.JSONBackedBenchmarkData(
            pathlib.Path("first.json"),
            {
                "commit": "123",
                "benchmarks": [{"benchmark_id": "first1"}, {"benchmark_id": "first2"}],
            },
        )

        second = benchmark_helper.JSONBackedBenchmarkData(
            pathlib.Path("second.json"),
            {
                "commit": "123",
                "benchmarks": [
                    {"benchmark_id": "second1"},
                    {"benchmark_id": "second2"},
                ],
            },
        )

        result = benchmark_helper.merge_results([first, second])

        self.assertEqual(
            result.data,
            {
                "commit": "123",
                "benchmarks": [
                    {"benchmark_id": "first1"},
                    {"benchmark_id": "first2"},
                    {"benchmark_id": "second1"},
                    {"benchmark_id": "second2"},
                ],
            },
        )

    def test_merge_results_mismatching_commits(self):
        first = benchmark_helper.JSONBackedBenchmarkData(
            pathlib.Path("first.json"), {"commit": "123", "benchmarks": []}
        )
        second = benchmark_helper.JSONBackedBenchmarkData(
            pathlib.Path("second.json"), {"commit": "456", "benchmarks": []}
        )

        with self.assertRaisesRegex(ValueError, "based on different commits"):
            benchmark_helper.merge_results([first, second])

    def test_create_json_backed_benchmark_data_success(self):
        benchmark_helper.JSONBackedBenchmarkData(
            pathlib.Path("first.json"), {"commit": "123", "benchmarks": []}
        )

    def test_create_json_backed_benchmark_data_with_missing_benchmark_list(self):
        with self.assertRaisesRegex(ValueError, "'benchmarks' field not found"):
            benchmark_helper.JSONBackedBenchmarkData(
                pathlib.Path("second.json"), {"commit": "123"}
            )

    def test_load_from_file_success(self):
        with tempfile.TemporaryDirectory() as dir:
            filepath = pathlib.Path(dir) / "first.json"
            contents = {"commit": "123", "benchmarks": []}
            filepath.write_text(json.dumps(contents))

            result = benchmark_helper.JSONBackedBenchmarkData.load_from_file(filepath)
            self.assertEqual(result.data, contents)
            self.assertEqual(result.source_filepath, filepath)

    def test_load_from_file_invalid_json(self):
        with tempfile.TemporaryDirectory() as dir:
            filepath = pathlib.Path(dir) / "first.json"
            filepath.write_text("bliblablub")

            with self.assertRaisesRegex(
                ValueError, "seems not to be a valid JSON file"
            ):
                benchmark_helper.JSONBackedBenchmarkData.load_from_file(filepath)


if __name__ == "__main__":
    unittest.main()
