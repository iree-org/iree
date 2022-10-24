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
    source_url="https://example.com/xyz_saved_model.tar.gz",
    entry_function="predict",
    input_types=["2xf32"])


class CommonArtifactsTest(unittest.TestCase):

  def test_generate_model_artifacts_root(self):
    models = [TFLITE_MODEL, TF_MODEL]
    parent_path = pathlib.PurePath("root", "models")

    artifacts_root = common_artifacts.generate_model_artifacts_root(
        parent_path=parent_path, models=models)

    self.assertEqual(
        artifacts_root,
        common_artifacts.ModelArtifactsRoot(
            model_artifact_map=collections.OrderedDict(
                [(TFLITE_MODEL.id,
                  common_artifacts.ModelArtifact(
                      model=TFLITE_MODEL,
                      file_path=parent_path /
                      f"{TFLITE_MODEL.id}_{TFLITE_MODEL.name}.tflite")),
                 (TF_MODEL.id,
                  common_artifacts.ModelArtifact(
                      model=TF_MODEL,
                      file_path=parent_path /
                      f"{TF_MODEL.id}_{TF_MODEL.name}"))])))


if __name__ == "__main__":
  unittest.main()
