# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Log extractors for IREE triage tools.

This package provides extractor infrastructure for analyzing logs from various
sources (CI, ctest, build logs) and extracting structured diagnostic information.
"""

from common.extractors.base import Extractor
from common.extractors.bazel_error import BazelErrorExtractor
from common.extractors.build_error import BuildErrorExtractor
from common.extractors.cmake_error import CMakeErrorExtractor
from common.extractors.codeql_error import CodeQLErrorExtractor
from common.extractors.ctest_error import CTestErrorExtractor
from common.extractors.infrastructure_flake import InfrastructureFlakeExtractor
from common.extractors.mlir_compiler import MLIRCompilerExtractor
from common.extractors.onnx_test import ONNXTestExtractor
from common.extractors.precommit import PrecommitErrorExtractor
from common.extractors.pytest_error import PytestErrorExtractor
from common.extractors.sanitizer import SanitizerExtractor

__all__ = [
    "Extractor",
    "BazelErrorExtractor",
    "BuildErrorExtractor",
    "CMakeErrorExtractor",
    "CodeQLErrorExtractor",
    "CTestErrorExtractor",
    "InfrastructureFlakeExtractor",
    "MLIRCompilerExtractor",
    "ONNXTestExtractor",
    "PrecommitErrorExtractor",
    "PytestErrorExtractor",
    "SanitizerExtractor",
]
