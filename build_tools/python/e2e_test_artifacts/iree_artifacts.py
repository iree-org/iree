## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of IREE artifacts."""

from dataclasses import dataclass
from typing import Sequence, OrderedDict
import collections
import pathlib

from e2e_test_artifacts import model_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions


@dataclass(frozen=True)
class ModuleDirectory(object):
  """IREE module directory that accommodates the module and related files."""
  module_path: pathlib.PurePath
  compile_config: iree_definitions.CompileConfig


@dataclass(frozen=True)
class ImportedModelArtifact(object):
  """IREE imported model artifact."""
  imported_model: iree_definitions.ImportedModel
  file_path: pathlib.PurePath


@dataclass(frozen=True)
class ModelDirectory(object):
  """IREE model directory that accommodates the modules from the same model."""
  imported_model_artifact: ImportedModelArtifact
  # Map of module directories, keyed by the assoicated compile config id.
  module_dir_map: OrderedDict[str, ModuleDirectory]


@dataclass(frozen=True)
class ArtifactsRoot(object):
  # Map of IREE model directories, keyed by model id.
  model_dir_map: OrderedDict[str, ModelDirectory]


def _get_imported_model_path(
    parent_path: pathlib.PurePath,
    imported_model: iree_definitions.ImportedModel,
    model_artifact: model_artifacts.ModelArtifact) -> pathlib.PurePath:
  model = imported_model.model
  if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
    # Uses the MLIR model directly.
    return model_artifact.file_path
  # Imported model path: <parent_path>/<model_name>.mlir
  return parent_path / f"{model.name}.mlir"


def _build_module_directory(
    parent_path: pathlib.PurePath,
    module_generation_config: iree_definitions.ModuleGenerationConfig
) -> ModuleDirectory:
  compile_config = module_generation_config.compile_config
  # IREE module dir: <parent_path>/<compile_config_id>
  dir_path = parent_path / compile_config.id
  # Module path: <parent_path>/<compile_config_id>/<model_name>.vmfb
  module_path = dir_path / f"{module_generation_config.imported_model.model.name}.vmfb"
  return ModuleDirectory(module_path=module_path, compile_config=compile_config)


def generate_artifacts_root(
    parent_path: pathlib.PurePath,
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
    model = imported_model.model
    # IREE model dir: <parent_path>/<model_id>_<model_name>
    model_dir_path = parent_path / f"{model.id}_{model.name}"

    module_dir_map = collections.OrderedDict()
    for config in grouped_generation_configs[model.id]:
      module_dir_map[config.compile_config.id] = _build_module_directory(
          parent_path=model_dir_path, module_generation_config=config)

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
