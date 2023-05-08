## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

from e2e_test_artifacts import model_artifacts
from e2e_test_artifacts.cmake_generator import model_rule_generator
from e2e_test_framework.definitions import common_definitions


class CommonGeneratorsTest(unittest.TestCase):

  def test_generate_model_rule_map(self):
    model_a = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    model_b = common_definitions.Model(
        id="5678",
        name="tf_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_MHLO_MLIR,
        source_url="https://example.com/xyz_mlirl",
        entry_function="predict",
        input_types=["2xf32"])
    root_path = pathlib.PurePath("model_root")

    rule_map = model_rule_generator.generate_model_rule_map(
        root_path=root_path, models=[model_a, model_b])

    self.assertEqual(list(rule_map.keys()), [model_a.id, model_b.id])
    self.assertEqual(rule_map[model_a.id].target_name, f"model-{model_a.id}")
    self.assertEqual(
        rule_map[model_a.id].file_path,
        model_artifacts.get_model_path(model=model_a, root_path=root_path))
    self.assertEqual(rule_map[model_b.id].target_name, f"model-{model_b.id}")
    self.assertEqual(
        rule_map[model_b.id].file_path,
        model_artifacts.get_model_path(model=model_b, root_path=root_path))


if __name__ == "__main__":
  unittest.main()
