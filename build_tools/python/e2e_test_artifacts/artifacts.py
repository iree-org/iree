## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of the e2e test artifacts."""

from dataclasses import dataclass
from typing import Sequence
import pathlib

from e2e_test_artifacts import common_artifacts, iree_artifacts
from e2e_test_framework.definitions import iree_definitions
import benchmark_suites.iree.benchmark_collections

MODEL_ARTIFACTS_ROOT = pathlib.PurePath("models")
IREE_ARTIFACTS_ROOT = pathlib.PurePath("iree")


@dataclass(frozen=True)
class ArtifactRoot(object):
  """Root artifact directory."""
  model_artifact_root: common_artifacts.ModelArtifactRoot
  iree_artifact_root: iree_artifacts.ArtifactRoot


def _generate_artifact_root(
    iree_module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig]
) -> ArtifactRoot:
  """Generates and unions directory structures from the configs."""

  model_artifact_factory = common_artifacts.ModelArtifactFactory(
      parent_path=MODEL_ARTIFACTS_ROOT)

  iree_artifact_root = iree_artifacts.generate_artifact_root(
      parent_path=IREE_ARTIFACTS_ROOT,
      model_artifact_factory=model_artifact_factory,
      module_generation_configs=iree_module_generation_configs)

  return ArtifactRoot(
      model_artifact_root=model_artifact_factory.generate_artifact_root(),
      iree_artifact_root=iree_artifact_root)


def generate_default_artifact_root() -> ArtifactRoot:
  """Generates artifacts from all configs."""

  (iree_benchmark_module_generation_configs,
   _) = benchmark_suites.iree.benchmark_collections.generate_benchmarks()
  # Unions all module generation config and dedups.
  iree_module_generation_configs = list(
      set(iree_benchmark_module_generation_configs))
  iree_module_generation_configs.sort(key=lambda config: config.get_id())

  return _generate_artifact_root(
      iree_module_generation_configs=iree_module_generation_configs)
