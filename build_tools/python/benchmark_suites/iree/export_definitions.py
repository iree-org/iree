## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Definitions of exported benchmark configs.

The definitions should only use primitive types to be JSON serializable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExecutionBenchmarkConfig(object):
  """E2E model run configs with metadata."""
  run_configs: Dict[str, Any]
  module_dir_paths: List[str]
  host_environment: Optional[Dict[str, str]]


@dataclass(frozen=True)
class CompilationBenchmarkConfig(object):
  """Module generation configs with metadata."""
  generation_configs: Dict[str, Any]
  module_dir_paths: List[str]
