# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A list of benchmarks and their similarity thresholds."""

import re

from dataclasses import dataclass
from enum import Enum


class ThresholdUnit(Enum):
  PERCENTAGE = "%"  # Percentage
  VALUE_NS = "ns"  # Absolute value in nanoseconds


@dataclass
class BenchmarkThreshold:
  """Similarity threshold for benchmarks matching a regular expression."""
  # A regular expression to match against the benchmark identifier.
  regex: re.Pattern
  # A threshold for computing the benchmark value average. Benchmark sample
  # values from consecutive runs and within the given range will be considered
  # as similar (with some noise). They will be used to compute the moving
  # average. The number will be interpreted according to the given unit.
  # What value to set depends on the noise range of the particular benchmark.
  threshold: int
  unit: ThresholdUnit

  def get_threshold_str(self):
    """Returns a string representation of the threshold."""
    if self.unit == ThresholdUnit.PERCENTAGE:
      return f"{self.threshold}%"
    return self.threshold


# A list of benchmarks and their similarity thresholds.
# Order matters here: if multiple regexes match a single benchmark, the first
# match is used.
BENCHMARK_THRESHOLDS = [
    # Fluctuating benchmarks on CPUs.
    BenchmarkThreshold(re.compile(r"^DeepLabV3.*big-core.*LLVM-CPU.* @ Pixel"),
                       20, ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(
        re.compile(r"^MobileBertSquad.*big-core.*LLVM-CPU-Sync @ Pixel-4"), 50,
        ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(re.compile(r"^MobileNetV2.*LLVM-CPU @ Pixel"), 15,
                       ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(
        re.compile(r"^MobileNetV3Small.*LLVM-CPU-Sync @ Pixel-6"), 20,
        ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(
        re.compile(r"^MobileNetV3Small.*big-core.*LLVM-CPU @ Pixel-6"), 20,
        ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(
        re.compile(r"^MobileNetV3Small.*little-core.*LLVM-CPU @ Pixel"), 20,
        ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(
        re.compile(r"^MobileSSD.*little-core.*LLVM-CPU.* @ Pixel-6"), 20,
        ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(re.compile(r"^PoseNet.*big-core.*LLVM-CPU.* @ Pixel-6"),
                       20, ThresholdUnit.PERCENTAGE),

    # Fluctuating benchmarks on GPUs.
    BenchmarkThreshold(
        re.compile(r"^MobileNetV3Small.*full-inference.*GPU-Mali"), 2 * 10**6,
        ThresholdUnit.VALUE_NS),

    # Benchmarks that complete around 10ms on GPUs; using percentage is not
    # suitable anymore.
    BenchmarkThreshold(re.compile(r"^DeepLabV3.*GPU-Mali"), 1 * 10**6,
                       ThresholdUnit.VALUE_NS),
    BenchmarkThreshold(re.compile(r"^MobileNet.*GPU"), 1 * 10**6,
                       ThresholdUnit.VALUE_NS),

    # Default threshold for all benchmarks: 5%.
    BenchmarkThreshold(re.compile(r".*"), 5, ThresholdUnit.PERCENTAGE),
]

COMPILATION_TIME_THRESHOLDS = [
    # Default threshold: 5%.
    BenchmarkThreshold(re.compile(r".*"), 5, ThresholdUnit.PERCENTAGE),
]

TOTAL_DISPATCH_SIZE_THRESHOLDS = [
    # Default threshold: 5%.
    BenchmarkThreshold(re.compile(r".*"), 5, ThresholdUnit.PERCENTAGE),
]

TOTAL_ARTIFACT_SIZE_THRESHOLDS = [
    # Default threshold: 5%.
    BenchmarkThreshold(re.compile(r".*"), 5, ThresholdUnit.PERCENTAGE),
]
