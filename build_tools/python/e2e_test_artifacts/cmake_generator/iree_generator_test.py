## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_artifacts import iree_artifacts
from e2e_test_artifacts.cmake_generator import common_generators, iree_generator
from e2e_test_framework.definitions import common_definitions, iree_definitions

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
LINALG_MODEL = common_definitions.Model(
    id="9012",
    name="linalg_m",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://example.com/xyz.mlir",
    entry_function="main",
    input_types=["3xf32"])


class IreeRuleBuilderTest(unittest.TestCase):

  def setUp(self):
    self._builder = iree_generator.IreeRuleBuilder(package_name="${package}")

  def test_build_model_import_rule_tflite(self):
    model_rule = common_generators.ModelRule(target_name="model-1234",
                                             file_path="root/models/x.tflite",
                                             cmake_rules=["abc"])
    output_file_path = pathlib.PurePath("root", "iree", TFLITE_MODEL.id,
                                        f"{TFLITE_MODEL.name}.mlir")

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=iree_definitions.ImportedModel(
            model=TFLITE_MODEL,
            dialect_type=iree_definitions.MLIRDialectType.TOSA),
        output_file_path=output_file_path)

    self.assertEqual(rule.target_name, f"iree-imported-model-{TFLITE_MODEL.id}")
    self.assertEqual(rule.output_file_path, str(output_file_path))

  def test_build_model_import_rule_linalg(self):
    model_rule = common_generators.ModelRule(target_name="model-5678",
                                             file_path="root/models/y.mlir",
                                             cmake_rules=["abc"])

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=iree_definitions.ImportedModel(
            model=LINALG_MODEL,
            dialect_type=iree_definitions.MLIRDialectType.LINALG),
        output_file_path=pathlib.PurePath(model_rule.file_path))

    self.assertEqual(rule.target_name, model_rule.target_name)
    self.assertEqual(rule.output_file_path, model_rule.file_path)

  def test_build_module_compile_rule(self):
    model_import_rule = iree_generator.IreeModelImportRule(
        target_name=f"iree-import-model-{TFLITE_MODEL.id}",
        output_file_path=f"root/iree/{TFLITE_MODEL.id}/1234.mlir",
        cmake_rules=["abc"])
    compile_config = iree_definitions.CompileConfig(
        id="compa",
        tags=["defaults"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                X86_64_CASCADELAKE,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    output_file_path = pathlib.PurePath("root", "iree", TFLITE_MODEL.id,
                                        compile_config.id,
                                        f"{TFLITE_MODEL.name}.vmfb")

    rule = self._builder.build_module_compile_rule(
        model_import_rule=model_import_rule,
        imported_model=iree_definitions.ImportedModel(
            model=TFLITE_MODEL,
            dialect_type=iree_definitions.MLIRDialectType.TOSA),
        compile_config=compile_config,
        output_file_path=output_file_path)

    self.assertEqual(rule.target_name,
                     f"iree-module-{TFLITE_MODEL.id}-{compile_config.id}")
    self.assertEqual(rule.output_module_path, str(output_file_path))


class IreeGeneratorTest(unittest.TestCase):

  def test_generate_rules(self):
    compile_config_a = iree_definitions.CompileConfig(
        id="config_a",
        tags=["defaults"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                X86_64_CASCADELAKE,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    compile_config_b = iree_definitions.CompileConfig(
        id="config_b",
        tags=["experimentals"],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_architecture=common_definitions.DeviceArchitecture.
                X86_64_CASCADELAKE,
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ])
    artifacts_root = iree_artifacts.ArtifactsRoot(
        model_dir_map=collections.OrderedDict({
            TFLITE_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(
                        imported_model=iree_definitions.ImportedModel(
                            model=TFLITE_MODEL,
                            dialect_type=iree_definitions.MLIRDialectType.TOSA),
                        file_path=pathlib.PurePath(
                            "iree", TFLITE_MODEL.id,
                            f"{TFLITE_MODEL.name}.mlir")),
                    module_dir_map=collections.OrderedDict({
                        compile_config_a.id:
                            iree_artifacts.
                            ModuleDirectory(module_path=pathlib.PurePath(
                                "iree", TFLITE_MODEL.id, compile_config_a.id,
                                f"{TFLITE_MODEL.name}.vmfb"),
                                            compile_config=compile_config_a),
                        compile_config_b.id:
                            iree_artifacts.
                            ModuleDirectory(module_path=pathlib.PurePath(
                                "iree", TFLITE_MODEL.id, compile_config_b.id,
                                f"{TFLITE_MODEL.name}.vmfb"),
                                            compile_config=compile_config_b)
                    })),
            TF_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(
                        imported_model=iree_definitions.ImportedModel(
                            model=TF_MODEL,
                            dialect_type=iree_definitions.MLIRDialectType.TOSA),
                        file_path=pathlib.PurePath("iree", TF_MODEL.id,
                                                   f"{TF_MODEL.name}.mlir")),
                    module_dir_map=collections.OrderedDict({
                        compile_config_a.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=pathlib.PurePath(
                                    "iree", TF_MODEL.id, compile_config_a.id,
                                    f"{TF_MODEL.name}.vmfb"),
                                compile_config=compile_config_a)
                    }))
        }))
    model_rule_map = {
        TFLITE_MODEL.id:
            common_generators.ModelRule(target_name=f"model-{TFLITE_MODEL.id}",
                                        file_path="root/models/x.tflite",
                                        cmake_rules=["abc"]),
        TF_MODEL.id:
            common_generators.ModelRule(target_name=f"model-{TF_MODEL.id}",
                                        file_path="root/models/y.tflite",
                                        cmake_rules=["abc"])
    }

    cmake_rules = iree_generator.generate_rules(
        package_name="${package}",
        root_path=pathlib.PurePath("root"),
        artifacts_root=artifacts_root,
        model_rule_map=model_rule_map)

    concated_cmake_rules = "\n".join(cmake_rules)
    self.assertRegex(concated_cmake_rules,
                     f"iree-imported-model-{TFLITE_MODEL.id}")
    self.assertRegex(concated_cmake_rules, f"iree-imported-model-{TF_MODEL.id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{TFLITE_MODEL.id}-{compile_config_a.id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{TFLITE_MODEL.id}-{compile_config_b.id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-module-{TF_MODEL.id}-{compile_config_a.id}")


if __name__ == "__main__":
  unittest.main()
