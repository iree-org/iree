## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of IREE artifacts."""

from dataclasses import dataclass
from typing import List, Sequence
import collections
import pathlib

from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_artifacts import common_artifacts


@dataclass(frozen=True)
class ImportedModelArtifact(object):
  imported_model: iree_definitions.ImportedModel
  file_path: pathlib.PurePath


@dataclass(frozen=True)
class ModuleArtifact(object):
  module_generation_config: iree_definitions.ModuleGenerationConfig
  file_path: pathlib.PurePath


@dataclass(frozen=True)
class ModuleDirectory(object):
  """IREE module directory that accommodates the module and related files."""
  dir_path: pathlib.PurePath
  module_artifact: ModuleArtifact


@dataclass(frozen=True)
class ModelDirectory(object):
  """IREE model directory that accommodates the modules from the same model."""
  dir_path: pathlib.PurePath
  imported_model_artifact: ImportedModelArtifact
  # Map of module directory, keyed by assoicated compile config id.
  module_dir_map: collections.OrderedDict[str, ModuleDirectory]


def _create_iree_imported_model_artifact(
    parent_path: pathlib.PurePath,
    imported_model: iree_definitions.ImportedModel,
    model_artifact: common_artifacts.ModelArtifact) -> ImportedModelArtifact:
  model = imported_model.source_model
  if model.source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
    # Directly use the MLIR model.
    file_path = model_artifact.file_path
  else:
    # Imported MLIR path: <parent_path>/<model_name>.mlir
    file_path = parent_path / f"{model.name}.mlir"
  return ImportedModelArtifact(imported_model=imported_model,
                               file_path=file_path)


def _create_iree_module_artifact(
    parent_path: pathlib.PurePath,
    module_generation_config: iree_definitions.ModuleGenerationConfig
) -> ModuleArtifact:
  # Module path: <parent_path>/<model_name>.vmfb
  file_path = parent_path / f"{module_generation_config.model.source_model.name}.vmfb"
  return ModuleArtifact(module_generation_config=module_generation_config,
                        file_path=file_path)


def generate_directory_structures(
    parent_path: pathlib.PurePath,
    model_artifact_factory: common_artifacts.ModelArtifactFactory,
    module_generation_configs: Sequence[iree_definitions.ModuleGenerationConfig]
) -> collections.OrderedDict[str, ModelDirectory]:
  """Generates IREE directory structure from module generation configs."""

  dep_imported_models = collections.OrderedDict(
      (config.model.source_model.id, config.model)
      for config in module_generation_configs)

  grouped_generation_configs = dict(
      (model_id, []) for model_id in dep_imported_models.keys())
  for config in module_generation_configs:
    grouped_generation_configs[config.model.source_model.id].append(config)

  model_dir_map = collections.OrderedDict()
  for imported_model in dep_imported_models.values():
    model = imported_model.source_model

    model_artifact = model_artifact_factory.create(model)

    # IREE model dir: <parent_path>/<model_id>_<model_name>
    dir_path = parent_path / f"{model.id}_{model.name}"
    imported_model_artifact = _create_iree_imported_model_artifact(
        parent_path=dir_path,
        imported_model=imported_model,
        model_artifact=model_artifact)

    module_dir_map = collections.OrderedDict()
    for config in grouped_generation_configs[model.id]:
      compile_config_id = config.compile_config.id
      module_subdir_path = dir_path / compile_config_id
      module_artifact = _create_iree_module_artifact(
          parent_path=module_subdir_path, module_generation_config=config)
      module_dir_map[compile_config_id] = ModuleDirectory(
          dir_path=module_subdir_path, module_artifact=module_artifact)

    model_dir_map[model.id] = ModelDirectory(
        dir_path=dir_path,
        imported_model_artifact=imported_model_artifact,
        module_dir_map=module_dir_map)

  return model_dir_map
