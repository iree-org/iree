## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of common artifacts."""

from dataclasses import dataclass
import collections
import pathlib
import urllib.parse

from e2e_test_framework.definitions import common_definitions

# Archive extensions used to pack models.
ARCHIVE_FILE_EXTENSIONS = [".tar", ".gz"]


@dataclass(frozen=True)
class ModelArtifact(object):
  model: common_definitions.Model
  file_path: pathlib.PurePath


@dataclass(frozen=True)
class ModelArtifactRoot(object):
  # Map of model artifacts, keyed by model id.
  model_artifact_map: collections.OrderedDict[str, ModelArtifact]


class ModelArtifactFactory(object):
  """Creates and collects model artifacts."""

  _parent_path: pathlib.PurePath
  _model_artifact_map: collections.OrderedDict[str, ModelArtifact]

  def __init__(self, parent_path: pathlib.PurePath):
    self._parent_path = parent_path
    self._model_artifact_map = collections.OrderedDict()

  def generate_artifact_root(self) -> ModelArtifactRoot:
    return ModelArtifactRoot(
        model_artifact_map=collections.OrderedDict(self._model_artifact_map))

  def create(self, model: common_definitions.Model) -> ModelArtifact:
    if model.id in self._model_artifact_map:
      artifact = self._model_artifact_map[model.id]
      if artifact.model != model:
        raise ValueError(f"Model mismatched: {model.id}.")
      return artifact

    model_url = urllib.parse.urlparse(model.source_url)
    # Drop the archive extensions.
    file_exts = pathlib.PurePath(model_url.path).suffixes
    while len(file_exts) > 0 and file_exts[-1] in ARCHIVE_FILE_EXTENSIONS:
      file_exts.pop()
    model_ext = "".join(file_exts)

    # Model path: <model_artifacts_root>/<model_id>_<model_name><model_ext>
    file_path = self._parent_path / f"{model.id}_{model.name}{model_ext}"
    artifact = ModelArtifact(model=model, file_path=file_path)
    self._model_artifact_map[model.id] = artifact
    return artifact
