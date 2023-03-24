## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helpers that generates paths for IREE artifacts."""

from typing import Dict, Iterable
import pathlib

from e2e_test_artifacts import model_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions

IREE_ARTIFACT_PREFIX = "iree"
MODULE_FILENAME = "module.vmfb"
SCHEDULING_STATS_FILENAME = "scheduling_stats.json"


def _get_model_prefix(imported_model: iree_definitions.ImportedModel) -> str:
  """Returns the model prefix for IREE artifacts. The common prefix helps group
  artifacts from the same model together for easier navigation.
  """
  model = imported_model.model
  # IREE model prefix: <iree_artifact_prefix>_<model_name>
  return f"{IREE_ARTIFACT_PREFIX}_{model.name}"


def get_imported_model_path(
    imported_model: iree_definitions.ImportedModel,
    root_path: pathlib.PurePath = pathlib.PurePath()
) -> pathlib.PurePath:
  """Returns the path of an IREE imported MLIR model. If the source model is
  in MLIR format, returns the path of source model.
  
  Args:
    imported_model: IREE model importing config.
    root_path: path of the root artifact directory, on which the returned path
      will base.
  Returns:
    Path of the imported model file.
  """
  model = imported_model.model
  if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
    # Uses the MLIR model directly.
    return model_artifacts.get_model_path(model=model, root_path=root_path)

  model_prefix = _get_model_prefix(imported_model)
  # Imported model path: <root_path>/<model_prefix>_<imported_model_id>.mlir
  return (root_path / f"{model_prefix}_{imported_model.composite_id}.mlir")


def get_module_dir_path(
    module_generation_config: iree_definitions.ModuleGenerationConfig,
    root_path: pathlib.PurePath = pathlib.PurePath()
) -> pathlib.PurePath:
  """Returns the path of an IREE module directory, which contains the compiled
  module and related flag files.
  
  Args:
    module_generation_config: IREE module generation config.
    root_path: path of the root artifact directory, on which the returned path
      will base.
  Returns:
    Path of the module directory.
  """
  model_prefix = _get_model_prefix(module_generation_config.imported_model)
  # Module dir path: <root_path>/<model_prefix>_module_<gen_config_id>
  return (root_path /
          f"{model_prefix}_module_{module_generation_config.composite_id}")


def get_dependent_model_map(
    module_generation_configs: Iterable[iree_definitions.ModuleGenerationConfig]
) -> Dict[str, common_definitions.Model]:
  """Returns an ordered map of the dependent models keyed by model id."""
  return dict((config.imported_model.model.id, config.imported_model.model)
              for config in module_generation_configs)
