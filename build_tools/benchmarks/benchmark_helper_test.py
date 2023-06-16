#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import json
import unittest
import benchmark_helper


class BenchmarkHelperTest(unittest.TestCase):
    def test_merge_results_simple(self):
        first = io.StringIO(
            json.dumps(
                {
                    "commit": "123",
                    "benchmarks": [
                        {"benchmark_id": "first1"},
                        {"benchmark_id": "first2"},
                    ],
                }
            )
        )
        setattr(first, "name", "first.json")

        second = io.StringIO(
            json.dumps(
                {
                    "commit": "123",
                    "benchmarks": [
                        {"benchmark_id": "second1"},
                        {"benchmark_id": "second2"},
                    ],
                }
            )
        )
        setattr(second, "name", "second.json")

        result = benchmark_helper.merge_results([first, second])

        self.assertEqual(
            result,
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
        first = io.StringIO(json.dumps({"commit": "123", "benchmarks": []}))
        setattr(first, "name", "first.json")

        second = io.StringIO(json.dumps({"commit": "456", "benchmarks": []}))
        setattr(second, "name", "second.json")

        with self.assertRaisesRegex(RuntimeError, "based on different commits"):
            benchmark_helper.merge_results([first, second])

    def test_merge_results_missing_benchmark_list(self):
        first = io.StringIO(json.dumps({"commit": "123", "benchmarks": []}))
        setattr(first, "name", "first.json")

        second = io.StringIO(json.dumps({"commit": "123"}))
        setattr(second, "name", "second.json")

        with self.assertRaisesRegex(RuntimeError, '"benchmarks" field not found'):
            benchmark_helper.merge_results([first, second])

    def test_merge_results_invalid_json(self):
        first = io.StringIO(json.dumps({"commit": "123", "benchmarks": []}))
        setattr(first, "name", "first.json")

        second = io.StringIO("bliblablub")
        setattr(second, "name", "second.notjson")

        with self.assertRaisesRegex(RuntimeError, "seems not to be a valid JSON file"):
            benchmark_helper.merge_results([first, second])


if __name__ == "__main__":
    unittest.main()
