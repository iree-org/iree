## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules to build common artifacts."""

import collections
from dataclasses import dataclass
import pathlib
from typing import Dict, List, Sequence
import urllib.parse

from e2e_test_artifacts import common_artifacts
import cmake_builder.rules
import e2e_test_artifacts.cmake_rule_generators.utils as cmake_rule_generators_utils


@dataclass
class ModelRule(cmake_rule_generators_utils.CMakeRule):
  target_name: str
  file_path: str
  cmake_rule: str

  def get_rule(self) -> str:
    return self.cmake_rule


def generate_model_rule_map(
    root_path: pathlib.PurePath,
    model_artifact_map: Dict[str, common_artifacts.ModelArtifact]
) -> Dict[str, ModelRule]:
  """Returns the model rules in an ordered map."""

  model_rules = collections.OrderedDict()
  for model_artifact in model_artifact_map.values():
    model = model_artifact.model

    # Model target: <package_name>-model-<model_id>
    target_name = f"model-{model.id}"
    model_path = str(root_path / model_artifact.file_path)

    model_url = urllib.parse.urlparse(model.source_url)
    if model_url.scheme == "https":
      cmake_rule = (f'# Fetch the model from "{model.source_url}"\n' +
                    cmake_builder.rules.build_iree_fetch_artifact(
                        target_name=target_name,
                        source_url=model.source_url,
                        output=model_path,
                        unpack=True))
    else:
      raise ValueError("Unsupported model url: {model.source_url}.")

    model_rules[model.id] = ModelRule(target_name=target_name,
                                      file_path=model_path,
                                      cmake_rule=cmake_rule)

  return model_rules
