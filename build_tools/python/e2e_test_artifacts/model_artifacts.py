## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of common artifacts."""

from dataclasses import dataclass
from typing import Sequence, OrderedDict
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
class ArtifactsRoot(object):
  # Map of model artifacts, keyed by model id.
  model_artifact_map: OrderedDict[str, ModelArtifact]


def generate_artifacts_root(
    parent_path: pathlib.PurePath,
    models: Sequence[common_definitions.Model]) -> ArtifactsRoot:
  """Generates model directory structure."""

  model_artifact_map = collections.OrderedDict()
  for model in models:
    if model.id in model_artifact_map:
      raise ValueError(f"Duplicate model {model.id}.")

    model_url = urllib.parse.urlparse(model.source_url)
    # Drop the archive extensions.
    file_exts = pathlib.PurePath(model_url.path).suffixes
    while len(file_exts) > 0 and file_exts[-1] in ARCHIVE_FILE_EXTENSIONS:
      file_exts.pop()
    model_ext = "".join(file_exts)

    # Model path: <model_artifacts_root>/<model_id>_<model_name><model_ext>
    file_path = parent_path / f"{model.id}_{model.name}{model_ext}"
    model_artifact_map[model.id] = ModelArtifact(model=model,
                                                 file_path=file_path)

  return ArtifactsRoot(model_artifact_map=model_artifact_map)
