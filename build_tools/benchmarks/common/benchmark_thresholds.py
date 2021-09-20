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
  PERCENTAGE = 0  # Percentage
  VALUE_MS = 1  # Absolute value in milliseconds


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
# Order matters here: if multiple regexes match a single benchmark, the first match is used.
BENCHMARK_THRESHOLDS = [
    # Unstable and noisy GPU benchmarks.
    BenchmarkThreshold(re.compile(r"^DeepLabV3.*GPU-Mali"), 90,
                       ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(re.compile(r"^MobileNetV3Small.*GPU-Mali"), 30,
                       ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(re.compile(r"^MobileSSD.*GPU-Mali"), 90,
                       ThresholdUnit.PERCENTAGE),
    BenchmarkThreshold(re.compile(r"^PoseNet.*GPU-Mali"), 90,
                       ThresholdUnit.PERCENTAGE),

    # Fast GPU benchmarks that complete around 10ms, so using percentage is not suitable.
    BenchmarkThreshold(re.compile(r"^MobileNetV3Small.*GPU-Adreno"), 2,
                       ThresholdUnit.VALUE_MS),

    # Default threshold for all benchmarks: 5%.
    BenchmarkThreshold(re.compile(r".*"), 5, ThresholdUnit.PERCENTAGE),
]
