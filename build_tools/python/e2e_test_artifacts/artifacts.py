## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of the e2e test artifacts."""

from dataclasses import dataclass
from typing import Dict, List, Sequence
import pathlib

from e2e_test_artifacts import common_artifacts, iree_artifacts
from e2e_test_framework.definitions import iree_definitions
import benchmark_suites.iree.benchmark_collections

MODEL_ARTIFACTS_ROOT = pathlib.PurePath("models")
IREE_ARTIFACTS_ROOT = pathlib.PurePath("iree")


@dataclass(frozen=True)
class RootDirectory(object):
  """Root artifact directory."""
  # Map of model artifact, keyed by model id.
  model_artifact_map: Dict[str, common_artifacts.ModelArtifact]
  # Map of IREE model directory, keyed by model id.
  iree_model_dir_map: Dict[str, iree_artifacts.ModelDirectory]


def generate_root_directory_structure(
    iree_module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig]
) -> RootDirectory:
  """Generates and unions directory structures from the configs."""

  model_artifact_factory = common_artifacts.ModelArtifactFactory(
      parent_path=MODEL_ARTIFACTS_ROOT)

  iree_model_subdirs = iree_artifacts.generate_directory_structures(
      parent_path=IREE_ARTIFACTS_ROOT,
      model_artifact_factory=model_artifact_factory,
      module_generation_configs=iree_module_generation_configs)

  return RootDirectory(
      model_artifact_map=model_artifact_factory.get_model_artifact_map(),
      iree_model_dir_map=iree_model_subdirs)


def generate_full_directory_structure() -> RootDirectory:
  """Generates artifacts from all configs."""

  (iree_benchmark_module_generation_configs,
   _) = benchmark_suites.iree.benchmark_collections.generate_benchmarks()
  # Unions all module generation config and dedups.
  iree_module_generation_configs = list(
      set(iree_benchmark_module_generation_configs))
  iree_module_generation_configs.sort(key=lambda config: config.get_id())

  return generate_root_directory_structure(
      iree_module_generation_configs=iree_module_generation_configs)
