## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of the e2e test artifacts."""

from dataclasses import dataclass
from typing import Sequence
import pathlib

from benchmark_suites.iree import benchmark_collections
from e2e_model_tests import benchmark_model_tests
from e2e_test_artifacts import model_artifacts, iree_artifacts
from e2e_test_framework.models import model_groups
from e2e_test_framework.definitions import common_definitions, iree_definitions

MODEL_ARTIFACTS_ROOT = pathlib.PurePath("models")
IREE_ARTIFACTS_ROOT = pathlib.PurePath("iree")


@dataclass(frozen=True)
class ArtifactsRoot(object):
  """Root artifact directory."""
  model_artifacts_root: model_artifacts.ArtifactsRoot
  iree_artifacts_root: iree_artifacts.ArtifactsRoot


def _generate_artifacts_root(
    models: Sequence[common_definitions.Model],
    iree_module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig]
) -> ArtifactsRoot:
  """Generates and unions directory structures from the configs."""

  model_artifacts_root = model_artifacts.generate_artifacts_root(
      parent_path=MODEL_ARTIFACTS_ROOT, models=models)

  iree_artifacts_root = iree_artifacts.generate_artifacts_root(
      parent_path=IREE_ARTIFACTS_ROOT,
      model_artifacts_root=model_artifacts_root,
      module_generation_configs=iree_module_generation_configs)

  return ArtifactsRoot(model_artifacts_root=model_artifacts_root,
                       iree_artifacts_root=iree_artifacts_root)


def generate_default_artifacts_root() -> ArtifactsRoot:
  """Generates artifacts from all configs."""

  # Unions and dedups all IREE module generation configs.
  iree_module_gen_config_set = set()
  iree_benchmark_gen_configs, _ = benchmark_collections.generate_benchmarks()
  iree_module_gen_config_set.update(iree_benchmark_gen_configs)
  iree_module_gen_config_set.update(
      test_config.module_generation_config
      for test_config in benchmark_model_tests.generate_model_tests())
  iree_module_gen_configs = list(iree_module_gen_config_set)
  iree_module_gen_configs.sort(key=lambda config: config.get_id())

  return _generate_artifacts_root(
      models=model_groups.ALL,
      iree_module_generation_configs=iree_module_gen_configs)
