# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A list of noisy benchmarks and their average thresholds."""

import re

NOISY_BENCHMARKS = [
    # (regular expression to match the benchmark name, average threshold)
    (re.compile(r"^PoseNet.*GPU-Mali-G77"), "100%"),
    (re.compile(r"^DeepLabV3.*GPU-Mali-G77"), "100%"),
    (re.compile(r"^MobileSSD.*GPU-Mali-G77"), "100%"),
]
