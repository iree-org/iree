## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules to build common artifacts."""

from dataclasses import dataclass
from typing import List, OrderedDict
import collections
import pathlib
import urllib.parse

from e2e_test_artifacts import model_artifacts
import cmake_builder.rules


@dataclass
class ModelRule(object):
  target_name: str
  file_path: str
  cmake_rules: List[str]


def generate_model_rule_map(
    root_path: pathlib.PurePath, artifacts_root: model_artifacts.ArtifactsRoot
) -> OrderedDict[str, ModelRule]:
  """Returns the model rules in an ordered map."""

  model_rules = collections.OrderedDict()
  for model_artifact in artifacts_root.model_artifact_map.values():
    model = model_artifact.model
    # Model target: <package_name>-model-<model_id>
    target_name = f"model-{model.id}"
    model_path = str(root_path / model_artifact.file_path)

    model_url = urllib.parse.urlparse(model.source_url)
    if model_url.scheme == "https":
      cmake_rules = [
          f'# Fetch the model from "{model.source_url}"\n' +
          cmake_builder.rules.build_iree_fetch_artifact(
              target_name=target_name,
              source_url=model.source_url,
              output=model_path,
              unpack=True)
      ]
    else:
      raise ValueError("Unsupported model url: {model.source_url}.")

    model_rules[model.id] = ModelRule(target_name=target_name,
                                      file_path=model_path,
                                      cmake_rules=cmake_rules)

  return model_rules
