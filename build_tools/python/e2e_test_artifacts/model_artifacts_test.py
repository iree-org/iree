## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_artifacts import model_artifacts, test_configs


class ModelArtifactsTest(unittest.TestCase):

  def test_generate_artifacts_root(self):
    models = [test_configs.TFLITE_MODEL, test_configs.TF_MODEL]
    parent_path = pathlib.PurePath("root", "models")

    artifacts_root = model_artifacts.generate_artifacts_root(
        parent_path=parent_path, models=models)

    self.assertEqual(
        artifacts_root,
        model_artifacts.
        ArtifactsRoot(model_artifact_map=collections.OrderedDict([
            (test_configs.TFLITE_MODEL.id,
             model_artifacts.ModelArtifact(
                 model=test_configs.TFLITE_MODEL,
                 file_path=parent_path /
                 f"{test_configs.TFLITE_MODEL.id}_{test_configs.TFLITE_MODEL.name}.tflite"
             )),
            (test_configs.TF_MODEL.id,
             model_artifacts.ModelArtifact(
                 model=test_configs.TF_MODEL,
                 file_path=parent_path /
                 f"{test_configs.TF_MODEL.id}_{test_configs.TF_MODEL.name}"))
        ])))


if __name__ == "__main__":
  unittest.main()
