## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_framework.definitions import iree_definitions
from e2e_test_artifacts import model_artifacts, iree_artifacts, test_configs


class IreeArtifactsTest(unittest.TestCase):

  def test_generate_artifacts_root(self):
    model_artifacts_root = model_artifacts.ArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            test_configs.TFLITE_MODEL.id:
                model_artifacts.ModelArtifact(model=test_configs.TFLITE_MODEL,
                                              file_path=pathlib.PurePath("x")),
            test_configs.TF_MODEL.id:
                model_artifacts.ModelArtifact(model=test_configs.TF_MODEL,
                                              file_path=pathlib.PurePath("y"))
        }))
    parent_path = pathlib.PurePath("root", "iree")

    artifacts_root = iree_artifacts.generate_artifacts_root(
        parent_path=parent_path,
        model_artifacts_root=model_artifacts_root,
        module_generation_configs=[
            iree_definitions.ModuleGenerationConfig(
                imported_model=test_configs.TFLITE_IMPORTED_MODEL,
                compile_config=test_configs.COMPILE_CONFIG_A),
            iree_definitions.ModuleGenerationConfig(
                imported_model=test_configs.TFLITE_IMPORTED_MODEL,
                compile_config=test_configs.COMPILE_CONFIG_B),
            iree_definitions.ModuleGenerationConfig(
                imported_model=test_configs.TF_IMPORTED_MODEL,
                compile_config=test_configs.COMPILE_CONFIG_B),
        ])

    expect_tflite_dir_path = (
        parent_path /
        f"{test_configs.TFLITE_MODEL.id}_{test_configs.TFLITE_MODEL.name}")
    expect_tflite_imported_model_artifact = iree_artifacts.ImportedModelArtifact(
        imported_model=test_configs.TFLITE_IMPORTED_MODEL,
        file_path=expect_tflite_dir_path /
        f"{test_configs.TFLITE_MODEL.name}.mlir")
    expect_tflite_module_dir_map = collections.OrderedDict({
        test_configs.COMPILE_CONFIG_A.id:
            iree_artifacts.ModuleDirectory(
                module_path=expect_tflite_dir_path /
                test_configs.COMPILE_CONFIG_A.id /
                f"{test_configs.TFLITE_MODEL.name}.vmfb",
                compile_config=test_configs.COMPILE_CONFIG_A),
        test_configs.COMPILE_CONFIG_B.id:
            iree_artifacts.ModuleDirectory(
                module_path=expect_tflite_dir_path /
                test_configs.COMPILE_CONFIG_B.id /
                f"{test_configs.TFLITE_MODEL.name}.vmfb",
                compile_config=test_configs.COMPILE_CONFIG_B),
    })
    expect_tf_dir_path = (
        parent_path /
        f"{test_configs.TF_MODEL.id}_{test_configs.TF_MODEL.name}")
    expect_tf_imported_model_artifact = iree_artifacts.ImportedModelArtifact(
        imported_model=test_configs.TF_IMPORTED_MODEL,
        file_path=expect_tf_dir_path / f"{test_configs.TF_MODEL.name}.mlir")
    expect_tf_module_dir_map = collections.OrderedDict({
        test_configs.COMPILE_CONFIG_B.id:
            iree_artifacts.ModuleDirectory(
                module_path=expect_tf_dir_path /
                test_configs.COMPILE_CONFIG_B.id /
                f"{test_configs.TF_MODEL.name}.vmfb",
                compile_config=test_configs.COMPILE_CONFIG_B),
    })
    self.assertEqual(
        artifacts_root,
        iree_artifacts.ArtifactsRoot(model_dir_map=collections.OrderedDict({
            test_configs.TFLITE_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=
                    expect_tflite_imported_model_artifact,
                    module_dir_map=expect_tflite_module_dir_map),
            test_configs.TF_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=expect_tf_imported_model_artifact,
                    module_dir_map=expect_tf_module_dir_map)
        })))

  def test_generate_artifacts_root_model_not_found(self):
    model_artifacts_root = model_artifacts.ArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            test_configs.TF_MODEL.id:
                model_artifacts.ModelArtifact(model=test_configs.TF_MODEL,
                                              file_path=pathlib.PurePath("y"))
        }))
    parent_path = pathlib.PurePath("root", "iree")

    self.assertRaises(
        ValueError, lambda: iree_artifacts.generate_artifacts_root(
            parent_path=parent_path,
            model_artifacts_root=model_artifacts_root,
            module_generation_configs=[
                iree_definitions.ModuleGenerationConfig(
                    imported_model=test_configs.TFLITE_IMPORTED_MODEL,
                    compile_config=test_configs.COMPILE_CONFIG_A)
            ]))


if __name__ == "__main__":
  unittest.main()
