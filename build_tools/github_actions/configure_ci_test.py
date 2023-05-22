#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import configure_ci

SORTED_DEFAULT_BENCHMARK_PRESETS_STR = ",".join(
    sorted(configure_ci.DEFAULT_BENCHMARK_PRESET_GROUP))


class GetBenchmarkPresetsTest(unittest.TestCase):

  def test_get_benchmark_presets_no_preset(self):
    presets_str = configure_ci.get_benchmark_presets(
        trailers={},
        labels=["unrelated-labels"],
        is_pr=True,
        is_llvm_integrate_pr=False)

    self.assertEqual(presets_str, "")

  def test_get_benchmark_presets_from_pr_labels(self):
    presets_str = configure_ci.get_benchmark_presets(
        trailers={},
        labels=["benchmarks:x86_64", "benchmarks:cuda"],
        is_pr=True,
        is_llvm_integrate_pr=False)

    self.assertEqual(presets_str, "cuda,x86_64")

  def test_get_benchmark_presets_from_trailers_and_labels(self):
    presets_str = configure_ci.get_benchmark_presets(
        trailers={"benchmark-extra": "android-cpu,android-gpu"},
        labels=["benchmarks:x86_64", "benchmarks:cuda"],
        is_pr=True,
        is_llvm_integrate_pr=False)

    self.assertEqual(presets_str, "android-cpu,android-gpu,cuda,x86_64")

  def test_get_benchmark_presets_from_default_group(self):
    presets_str = configure_ci.get_benchmark_presets(
        trailers={"benchmark-extra": "default"},
        labels=[],
        is_pr=True,
        is_llvm_integrate_pr=False)

    self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)
    # Sanity check to ensure no `*-long` preset in the default group.
    self.assertNotIn("-long", presets_str)

  def test_get_benchmark_presets_for_non_pr(self):
    presets_str = configure_ci.get_benchmark_presets(trailers={},
                                                     labels=[],
                                                     is_pr=False,
                                                     is_llvm_integrate_pr=False)

    self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)

  def test_get_benchmark_presets_for_llvm_integrate_pr(self):
    presets_str = configure_ci.get_benchmark_presets(trailers={},
                                                     labels=[],
                                                     is_pr=True,
                                                     is_llvm_integrate_pr=True)

    self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)

  # Sample PR description:
  # ```
  # PR Title
  #
  # PR body...
  #
  # skip-llvm-integrate-benchmark: some good reasons
  # ```
  # Result: No benchmark is automatically enabled on the LLVM integrate PR.
  def test_get_benchmark_presets_skip_llvm_integrate_benchmark(self):
    presets_str = configure_ci.get_benchmark_presets(
        trailers={"skip-llvm-integrate-benchmark": "some good reasons"},
        labels=[],
        is_pr=True,
        is_llvm_integrate_pr=True)

    self.assertEqual(presets_str, "")

  def test_get_benchmark_presets_unknown_preset(self):
    self.assertRaises(
        ValueError, lambda: configure_ci.get_benchmark_presets(
            trailers={"benchmark-extra": "unknown"},
            labels=[],
            is_pr=True,
            is_llvm_integrate_pr=False))


if __name__ == "__main__":
  unittest.main()
