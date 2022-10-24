## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_artifacts import common_artifacts, test_configs
from e2e_test_artifacts.cmake_generator import common_generators


class CommonGeneratorsTest(unittest.TestCase):

  def test_generate_model_rule_map(self):
    artifacts_root = common_artifacts.ModelArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            test_configs.TFLITE_MODEL.id:
                common_artifacts.ModelArtifact(model=test_configs.TFLITE_MODEL,
                                               file_path=pathlib.PurePath(
                                                   "1234.tflite")),
            test_configs.TF_MODEL.id:
                common_artifacts.ModelArtifact(model=test_configs.TF_MODEL,
                                               file_path=pathlib.PurePath(
                                                   "5678_saved_model"))
        }))
    root_path = pathlib.PurePath("model_root")

    rule_map = common_generators.generate_model_rule_map(
        root_path=root_path, artifacts_root=artifacts_root)

    self.assertEqual(list(rule_map.keys()),
                     [test_configs.TFLITE_MODEL.id, test_configs.TF_MODEL.id])
    self.assertEqual(rule_map[test_configs.TFLITE_MODEL.id].target_name,
                     f"model-{test_configs.TFLITE_MODEL.id}")
    self.assertEqual(rule_map[test_configs.TFLITE_MODEL.id].file_path,
                     str(root_path / "1234.tflite"))
    self.assertEqual(rule_map[test_configs.TF_MODEL.id].target_name,
                     f"model-{test_configs.TF_MODEL.id}")
    self.assertEqual(rule_map[test_configs.TF_MODEL.id].file_path,
                     str(root_path / "5678_saved_model"))


if __name__ == "__main__":
  unittest.main()
