## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_artifacts import model_artifacts, iree_artifacts


class IreeArtifactsTest(unittest.TestCase):

  def test_get_imported_model_path(self):
    model = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    imported_model = iree_definitions.ImportedModel.from_model(model)
    root_path = pathlib.PurePath("root")

    path = iree_artifacts.get_imported_model_path(imported_model=imported_model,
                                                  root_path=root_path)

    self.assertEqual(
        path, root_path / f"{iree_artifacts.IREE_ARTIFACT_PREFIX}_{model.name}_"
        f"{imported_model.composite_id}.mlir")

  def test_get_imported_model_path_with_mlir_model(self):
    model = common_definitions.Model(
        id="9012",
        name="linalg_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
        source_url="https://example.com/xyz.mlir",
        entry_function="main",
        input_types=["3xf32"])
    imported_model = iree_definitions.ImportedModel.from_model(model)
    root_path = pathlib.PurePath("root")

    path = iree_artifacts.get_imported_model_path(imported_model=imported_model,
                                                  root_path=root_path)

    self.assertEqual(
        path, model_artifacts.get_model_path(model=model, root_path=root_path))

  def test_get_module_dir_path(self):
    model = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    imported_model = iree_definitions.ImportedModel.from_model(model)
    compile_config = iree_definitions.CompileConfig.build(
        id="config_a",
        tags=["defaults"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                X86_64_CASCADELAKE,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    gen_config = iree_definitions.ModuleGenerationConfig.build(
        imported_model=imported_model, compile_config=compile_config)
    root_path = pathlib.PurePath("root")

    path = iree_artifacts.get_module_dir_path(
        module_generation_config=gen_config, root_path=root_path)

    self.assertEqual(
        path, root_path / f"{iree_artifacts.IREE_ARTIFACT_PREFIX}_{model.name}_"
        f"module_{gen_config.composite_id}")

  def test_get_dependent_model_map(self):
    model_a = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    model_b = common_definitions.Model(
        id="9012",
        name="linalg_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
        source_url="https://example.com/xyz.mlir",
        entry_function="main",
        input_types=["3xf32"])
    imported_model_a = iree_definitions.ImportedModel.from_model(model_a)
    imported_model_b = iree_definitions.ImportedModel.from_model(model_b)
    compile_config_a = iree_definitions.CompileConfig.build(
        id="config_a",
        tags=["defaults"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                X86_64_CASCADELAKE,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    compile_config_b = iree_definitions.CompileConfig.build(
        id="config_b",
        tags=["defaults"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                RV64_GENERIC,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    gen_config_a = iree_definitions.ModuleGenerationConfig.build(
        imported_model=imported_model_a, compile_config=compile_config_a)
    gen_config_b = iree_definitions.ModuleGenerationConfig.build(
        imported_model=imported_model_b, compile_config=compile_config_a)
    gen_config_c = iree_definitions.ModuleGenerationConfig.build(
        imported_model=imported_model_b, compile_config=compile_config_b)

    models = iree_artifacts.get_dependent_model_map(
        module_generation_configs=[gen_config_a, gen_config_b, gen_config_c])

    self.assertEqual(models, {model_a.id: model_a, model_b.id: model_b})


if __name__ == "__main__":
  unittest.main()
