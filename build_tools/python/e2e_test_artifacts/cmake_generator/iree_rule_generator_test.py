## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

from e2e_test_artifacts.cmake_generator import model_rule_generator, iree_rule_generator
from e2e_test_framework.definitions import common_definitions, iree_definitions


class IreeRuleBuilderTest(unittest.TestCase):

  def setUp(self):
    self._builder = iree_rule_generator.IreeRuleBuilder(
        package_name="${package}")

  def test_build_model_import_rule_tflite(self):
    tflite_model = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32"])
    tflite_imported_model = iree_definitions.ImportedModel.from_model(
        tflite_model)
    model_rule = model_rule_generator.ModelRule(
        target_name="model-1234",
        file_path=pathlib.PurePath("root/models/x.tflite"),
        cmake_rules=["abc"])
    output_file_path = pathlib.PurePath("root", "iree", tflite_model.id,
                                        f"{tflite_model.name}.mlir")

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=tflite_imported_model,
        output_file_path=output_file_path)

    self.assertEqual(
        rule.target_name,
        f"iree-imported-model-{tflite_imported_model.composite_id}")
    self.assertEqual(rule.output_file_path, output_file_path)

  def test_build_model_import_rule_linalg(self):
    linalg_model = common_definitions.Model(
        id="9012",
        name="linalg_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
        source_url="https://example.com/xyz.mlir",
        entry_function="main",
        input_types=["3xf32"])
    linalg_imported_model = iree_definitions.ImportedModel.from_model(
        linalg_model)
    model_rule = model_rule_generator.ModelRule(
        target_name="model-5678",
        file_path=pathlib.PurePath("root/models/y.mlir"),
        cmake_rules=["abc"])

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=linalg_imported_model,
        output_file_path=pathlib.PurePath(model_rule.file_path))

    self.assertEqual(rule.target_name, model_rule.target_name)
    self.assertEqual(pathlib.PurePath(rule.output_file_path),
                     pathlib.PurePath(model_rule.file_path))

  def test_build_module_compile_rule(self):
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
    model_import_rule = iree_rule_generator.IreeModelImportRule(
        target_name=f"iree-import-model-abcd",
        output_file_path=pathlib.PurePath("root/iree/abcd/1234.mlir"),
        cmake_rules=["abc"])
    output_file_path = pathlib.PurePath("root/iree/test_output")

    rule = self._builder.build_module_compile_rule(
        model_import_rule=model_import_rule,
        module_generation_config=gen_config,
        output_file_path=output_file_path)

    self.assertEqual(rule.target_name, f"iree-module-{gen_config.composite_id}")
    self.assertEqual(rule.output_module_path, output_file_path)

  def test_build_target_path(self):
    builder = iree_rule_generator.IreeRuleBuilder(package_name="xyz")

    path = builder.build_target_path("target-abc")

    self.assertEqual(path, f"xyz_target-abc")


class IreeGeneratorTest(unittest.TestCase):

  def test_generate_rules(self):
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
        name="stablehlo_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
        source_url="https://example.com/xyz_stablehlo.mlir",
        entry_function="predict",
        input_types=["2xf32"])
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
    model_rule_map = {
        model_a.id:
            model_rule_generator.ModelRule(
                target_name=f"model-x",
                file_path=pathlib.PurePath("x.tflite"),
                cmake_rules=["abc"]),
        model_b.id:
            model_rule_generator.ModelRule(
                target_name=f"model-y",
                file_path=pathlib.PurePath("root/model_5678_stablehlo_m.mlir"),
                cmake_rules=["efg"]),
    }

    cmake_rules = iree_rule_generator.generate_rules(
        package_name="${package}",
        root_path=pathlib.PurePath("root"),
        module_generation_configs=[gen_config_a, gen_config_b, gen_config_c],
        model_rule_map=model_rule_map)

    concated_cmake_rules = "\n".join(cmake_rules)
    self.assertRegex(concated_cmake_rules,
                     f"iree-imported-model-{imported_model_a.composite_id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{gen_config_a.composite_id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{gen_config_b.composite_id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{gen_config_c.composite_id}")


if __name__ == "__main__":
  unittest.main()
