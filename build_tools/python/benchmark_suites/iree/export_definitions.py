## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Definitions related to export benchmark configs.

The definitions only use primitive types so they are JSON serializable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExecutionBenchmarkConfig(object):
  host_environment: Dict[str, str]
  module_dir_paths: List[str]
  run_configs: Dict[str, Any]


@dataclass(frozen=True)
class CompilationBenchmarkConfig(object):
  module_dir_paths: List[str]
  generation_configs: Dict[str, Any]
