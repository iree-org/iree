# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A list of noisy benchmarks and their average thresholds."""

import re

# A list of noisy benchmarks. Each one is a tuple that contains the following
# fields:
# - A regular expression to match against the benchmark identifier.
# - A threshold for computing the benchmark value average. Benchmark sample
#   values from consecutive runs and within the given range will be considered
#   as similar (with some noise). They will be used to compute the moving
#   average. The number will be interpreted as a percentage. What value to set
#   depends on the noise range of the particular benchmark.
NOISY_BENCHMARKS = [
    (re.compile(r"^DeepLabV3.*GPU-Mali-G77"), 100),
    (re.compile(r"^MobileSSD.*GPU-Mali-G77"), 100),
    (re.compile(r"^PoseNet.*GPU-Mali-G77"), 100),
]
