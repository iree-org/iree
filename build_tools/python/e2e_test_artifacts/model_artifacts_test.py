## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

from e2e_test_artifacts import model_artifacts
from e2e_test_framework.definitions import common_definitions


class ModelArtifactsTest(unittest.TestCase):

  def test_get_model_path_with_tflite_model(self):
    tflite_model = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    root_path = pathlib.PurePath("root")

    path = model_artifacts.get_model_path(model=tflite_model,
                                          root_path=root_path)

    self.assertEqual(
        path, root_path /
        f"{model_artifacts.MODEL_ARTIFACT_PREFIX}_{tflite_model.id}_{tflite_model.name}.tflite"
    )

  def test_get_model_path_with_tf_model(self):
    tf_model = common_definitions.Model(
        id="5678",
        name="tf_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
        source_url="https://example.com/xyz_mlir",
        entry_function="predict",
        input_types=["2xf32"])
    root_path = pathlib.PurePath("root")

    path = model_artifacts.get_model_path(model=tf_model, root_path=root_path)

    self.assertEqual(
        path, root_path /
        f"{model_artifacts.MODEL_ARTIFACT_PREFIX}_{tf_model.id}_{tf_model.name}"
    )


if __name__ == "__main__":
  unittest.main()
