## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_artifacts import common_artifacts, iree_artifacts

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
COMPILE_TARGET = iree_definitions.CompileTarget(
    target_architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
    target_backend=iree_definitions.TargetBackend.LLVM_CPU,
    target_abi=iree_definitions.TargetABI.LINUX_ANDROID29)


class IreeArtifactsTest(unittest.TestCase):

  def test_generate_artifacts_root(self):
    model_artifacts_root = common_artifacts.ModelArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            TFLITE_MODEL.id:
                common_artifacts.ModelArtifact(model=TFLITE_MODEL,
                                               file_path=pathlib.PurePath("x")),
            TF_MODEL.id:
                common_artifacts.ModelArtifact(model=TF_MODEL,
                                               file_path=pathlib.PurePath("y"))
        }))
    compile_config = iree_definitions.CompileConfig(
        id="a", tags=["a"], compile_targets=[COMPILE_TARGET])
    tflite_imported_model = iree_definitions.ImportedModel(
        model=TFLITE_MODEL, dialect_type=iree_definitions.MLIRDialectType.TOSA)
    tf_imported_model = iree_definitions.ImportedModel(
        model=TF_MODEL, dialect_type=iree_definitions.MLIRDialectType.MHLO)
    parent_path = pathlib.PurePath("root", "iree")

    artifacts_root = iree_artifacts.generate_artifacts_root(
        parent_path=parent_path,
        model_artifacts_root=model_artifacts_root,
        module_generation_configs=[
            iree_definitions.ModuleGenerationConfig(
                imported_model=tflite_imported_model,
                compile_config=compile_config),
            iree_definitions.ModuleGenerationConfig(
                imported_model=tf_imported_model, compile_config=compile_config)
        ])

    self.assertEqual(
        artifacts_root,
        iree_artifacts.ArtifactsRoot(model_dir_map=collections.OrderedDict({
            TFLITE_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(
                        imported_model=tflite_imported_model,
                        file_path=parent_path /
                        f"{TFLITE_MODEL.id}_{TFLITE_MODEL.name}" /
                        f"{TFLITE_MODEL.name}.mlir"),
                    module_dir_map=collections.OrderedDict({
                        compile_config.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=parent_path /
                                f"{TFLITE_MODEL.id}_{TFLITE_MODEL.name}" /
                                compile_config.id / f"{TFLITE_MODEL.name}.vmfb",
                                compile_config=compile_config),
                    })),
            TF_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(imported_model=tf_imported_model,
                                          file_path=parent_path /
                                          f"{TF_MODEL.id}_{TF_MODEL.name}" /
                                          f"{TF_MODEL.name}.mlir"),
                    module_dir_map=collections.OrderedDict({
                        compile_config.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=parent_path /
                                f"{TF_MODEL.id}_{TF_MODEL.name}" /
                                compile_config.id / f"{TF_MODEL.name}.vmfb",
                                compile_config=compile_config)
                    }))
        })))

  def test_generate_artifacts_root_model_not_found(self):
    model_artifacts_root = common_artifacts.ModelArtifactsRoot(
        model_artifact_map=collections.OrderedDict({
            TFLITE_MODEL.id:
                common_artifacts.ModelArtifact(model=TFLITE_MODEL,
                                               file_path=pathlib.PurePath("x"))
        }))
    compile_config = iree_definitions.CompileConfig(
        id="a", tags=["a"], compile_targets=[COMPILE_TARGET])
    tf_imported_model = iree_definitions.ImportedModel(
        model=TF_MODEL, dialect_type=iree_definitions.MLIRDialectType.MHLO)
    parent_path = pathlib.PurePath("root", "iree")

    self.assertRaises(
        ValueError, lambda: iree_artifacts.generate_artifacts_root(
            parent_path=parent_path,
            model_artifacts_root=model_artifacts_root,
            module_generation_configs=[
                iree_definitions.ModuleGenerationConfig(
                    imported_model=tf_imported_model,
                    compile_config=compile_config)
            ]))


if __name__ == "__main__":
  unittest.main()
