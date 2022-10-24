## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_framework.definitions import common_definitions
from e2e_test_artifacts import common_artifacts
from e2e_test_artifacts.cmake_generator import common_generators

TFLITE_MODEL = common_definitions.Model(
    id="1234",
    name="tflite_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    source_url="https://example.com/xyz.tflite",
    entry_function="main",
    input_types=["1xf32"])
TF_MODEL = common_definitions.Model(
    id="5678",
    name="tf_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF,
    source_url="https://example.com/xyz_saved_model",
    entry_function="predict",
    input_types=["2xf32"])


class CommonGeneratorsTest(unittest.TestCase):

  def test_generate_model_rule_map(self):
    artifacts_root = common_artifacts.ModelArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            TFLITE_MODEL.id:
                common_artifacts.ModelArtifact(model=TFLITE_MODEL,
                                               file_path=pathlib.PurePath(
                                                   "models", "1234.tflite")),
            TF_MODEL.id:
                common_artifacts.ModelArtifact(model=TF_MODEL,
                                               file_path=pathlib.PurePath(
                                                   "models", "5678_model"))
        }))

    rule_map = common_generators.generate_model_rule_map(
        root_path=pathlib.PurePath("root"), artifacts_root=artifacts_root)

    self.assertEqual(list(rule_map.keys()), [TFLITE_MODEL.id, TF_MODEL.id])
    self.assertEqual(rule_map[TFLITE_MODEL.id].target_name,
                     f"model-{TFLITE_MODEL.id}")
    self.assertEqual(rule_map[TFLITE_MODEL.id].file_path,
                     str(pathlib.PurePath("root", "models", "1234.tflite")))
    self.assertEqual(rule_map[TF_MODEL.id].target_name, f"model-{TF_MODEL.id}")
    self.assertEqual(rule_map[TF_MODEL.id].file_path,
                     str(pathlib.PurePath("root", "models", "5678_model")))


if __name__ == "__main__":
  unittest.main()
