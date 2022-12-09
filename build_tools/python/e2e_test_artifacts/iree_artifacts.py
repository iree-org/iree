## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of IREE artifacts."""

from typing import Sequence
import collections
import pathlib

from e2e_test_artifacts import model_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions

IREE_ARTIFACT_PREFIX = "iree"
MODULE_FILENAME = "module.vmfb"


def _get_model_prefix(imported_model: iree_definitions.ImportedModel) -> str:
  """Returns the path of an IREE model dir."""
  model = imported_model.model
  # IREE model prefix: <iree_artifact_prefix>_<model_id>_<model_name>
  return f"{IREE_ARTIFACT_PREFIX}_{model.id}_{model.name}"


def get_imported_model_path(
    imported_model: iree_definitions.ImportedModel,
    root_dir: pathlib.PurePath = pathlib.PurePath()
) -> pathlib.PurePath:
  """Returns the path of an IREE imported MLIR model. If the source model is
  in MLIR format, returns the path of source model.
  
  Args:
    imported_model: IREE model importing config.
    root_dir: path of the root artifact directory, on which the returned path
      will base.
  Returns:
    Path of the imported model file.
  """
  model = imported_model.model
  if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
    # Uses the MLIR model directly.
    return model_artifacts.get_model_path(model=model, root_dir=root_dir)

  model_prefix = _get_model_prefix(imported_model)
  # Imported model path: <root_dir>/<model_prefix>.mlir
  return root_dir / f"{model_prefix}.mlir"


def get_module_dir_path(
    module_generation_config: iree_definitions.ModuleGenerationConfig,
    root_dir: pathlib.PurePath = pathlib.PurePath()
) -> pathlib.PurePath:
  """Returns the path of an IREE module directory, which contains the compiled
  module and related flag files.
  
  Args:
    module_generation_config: IREE module generation config.
    root_dir: path of the root artifact directory, on which the returned path
      will base.
  Returns:
    Path of the module directory.
  """
  model_prefix = _get_model_prefix(module_generation_config.imported_model)
  # Module path: <root_dir>/<model_prefix>_<compile_config_id>
  return root_dir / f"{model_prefix}_{module_generation_config.compile_config.id}"


def generate_artifacts_root(
    root_dir: pathlib.PurePath,
    model_artifacts_root: model_artifacts.ArtifactsRoot,
    module_generation_configs: Sequence[iree_definitions.ModuleGenerationConfig]
) -> ArtifactsRoot:
  """Generates IREE directory structure from module generation configs."""

  all_imported_models = collections.OrderedDict(
      (config.imported_model.model.id, config.imported_model)
      for config in module_generation_configs)

  grouped_generation_configs = dict(
      (model_id, []) for model_id in all_imported_models.keys())
  for config in module_generation_configs:
    grouped_generation_configs[config.imported_model.model.id].append(config)

  model_dir_map = collections.OrderedDict()
  for imported_model in all_imported_models.values():
    model_dir_path = _get_model_dir_path(root_dir=root_dir,
                                         imported_model=imported_model)
    model = imported_model.model

    module_dir_map = collections.OrderedDict()
    for config in grouped_generation_configs[model.id]:
      module_dir_map[config.compile_config.id] = _build_module_directory(
          root_dir=root_dir, module_generation_config=config)

    model_artifact = model_artifacts_root.model_artifact_map.get(model.id)
    if model_artifact is None:
      raise ValueError(f"Model artifact {model.id} not found.")

    imported_model_path = _get_imported_model_path(
        parent_path=model_dir_path,
        imported_model=imported_model,
        model_artifact=model_artifact)

    model_dir_map[model.id] = ModelDirectory(
        imported_model_artifact=ImportedModelArtifact(
            imported_model=imported_model, file_path=imported_model_path),
        module_dir_map=module_dir_map)

  return ArtifactsRoot(model_dir_map=model_dir_map)
