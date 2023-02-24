## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake rules to fetch model artifacts."""

from dataclasses import dataclass
from typing import Dict, Iterable, List
import pathlib
import urllib.parse

from e2e_test_artifacts import model_artifacts
from e2e_test_framework.definitions import common_definitions
import cmake_builder.rules


@dataclass
class ModelRule(object):
  target_name: str
  file_path: pathlib.PurePath
  cmake_rules: List[str]


def generate_model_rule_map(
    root_path: pathlib.PurePath,
    models: Iterable[common_definitions.Model]) -> Dict[str, ModelRule]:
  """Returns the model rules keyed by model id in an ordered map."""

  model_rules = {}
  for model in models:
    # Model target: <package_name>-model-<model_id>
    target_name = cmake_builder.rules.sanitize_target_name(f"model-{model}")
    model_path = model_artifacts.get_model_path(model=model,
                                                root_path=root_path)

    model_url = urllib.parse.urlparse(model.source_url)
    if model_url.scheme == "https":
      cmake_rules = [
          cmake_builder.rules.build_iree_fetch_artifact(
              target_name=target_name,
              source_url=model.source_url,
              output=str(model_path),
              unpack=True)
      ]
    else:
      raise ValueError("Unsupported model url: {model.source_url}.")

    model_rules[model.id] = ModelRule(target_name=target_name,
                                      file_path=model_path,
                                      cmake_rules=cmake_rules)

  return model_rules
