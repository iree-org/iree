## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import pathlib
import unittest

from e2e_test_artifacts import iree_artifacts, test_configs
from e2e_test_artifacts.cmake_generator import model_rule_generator, iree_rule_generator


class IreeRuleBuilderTest(unittest.TestCase):

  def setUp(self):
    self._builder = iree_rule_generator.IreeRuleBuilder(
        package_name="${package}")

  def test_build_model_import_rule_tflite(self):
    model_rule = model_rule_generator.ModelRule(
        target_name="model-1234",
        file_path="root/models/x.tflite",
        cmake_rules=["abc"])
    output_file_path = pathlib.PurePath(
        "root", "iree", test_configs.TFLITE_MODEL.id,
        f"{test_configs.TFLITE_MODEL.name}.mlir")

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=test_configs.TFLITE_IMPORTED_MODEL,
        output_file_path=output_file_path)

    self.assertEqual(rule.target_name,
                     f"iree-imported-model-{test_configs.TFLITE_MODEL.id}")
    self.assertEqual(rule.output_file_path, str(output_file_path))

  def test_build_model_import_rule_linalg(self):
    model_rule = model_rule_generator.ModelRule(target_name="model-5678",
                                                file_path="root/models/y.mlir",
                                                cmake_rules=["abc"])

    rule = self._builder.build_model_import_rule(
        source_model_rule=model_rule,
        imported_model=test_configs.LINALG_IMPORTED_MODEL,
        output_file_path=pathlib.PurePath(model_rule.file_path))

    self.assertEqual(rule.target_name, model_rule.target_name)
    self.assertEqual(pathlib.PurePath(rule.output_file_path),
                     pathlib.PurePath(model_rule.file_path))

  def test_build_module_compile_rule(self):
    model_import_rule = iree_rule_generator.IreeModelImportRule(
        target_name=f"iree-import-model-abcd",
        output_file_path=f"root/iree/abcd/1234.mlir",
        cmake_rules=["abc"])
    output_file_path = pathlib.PurePath("root/iree/test_output")

    rule = self._builder.build_module_compile_rule(
        model_import_rule=model_import_rule,
        imported_model=test_configs.TFLITE_IMPORTED_MODEL,
        compile_config=test_configs.COMPILE_CONFIG_A,
        output_file_path=output_file_path)

    self.assertEqual(
        rule.target_name,
        f"iree-module-{test_configs.TFLITE_MODEL.id}-{test_configs.COMPILE_CONFIG_A.id}"
    )
    self.assertEqual(rule.output_module_path, str(output_file_path))


class IreeGeneratorTest(unittest.TestCase):

  def test_generate_rules(self):
    artifacts_root = iree_artifacts.ArtifactsRoot(
        model_dir_map=collections.OrderedDict({
            test_configs.TFLITE_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(
                        imported_model=test_configs.TFLITE_IMPORTED_MODEL,
                        file_path=pathlib.PurePath("1234")),
                    module_dir_map=collections.OrderedDict({
                        test_configs.COMPILE_CONFIG_A.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=pathlib.PurePath("abcd"),
                                compile_config=test_configs.COMPILE_CONFIG_A),
                        test_configs.COMPILE_CONFIG_B.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=pathlib.PurePath("efgh"),
                                compile_config=test_configs.COMPILE_CONFIG_B)
                    })),
            test_configs.TF_MODEL.id:
                iree_artifacts.ModelDirectory(
                    imported_model_artifact=iree_artifacts.
                    ImportedModelArtifact(
                        imported_model=test_configs.TF_IMPORTED_MODEL,
                        file_path=pathlib.PurePath("5678")),
                    module_dir_map=collections.OrderedDict({
                        test_configs.COMPILE_CONFIG_A.id:
                            iree_artifacts.ModuleDirectory(
                                module_path=pathlib.PurePath("xyz"),
                                compile_config=test_configs.COMPILE_CONFIG_A)
                    }))
        }))
    model_rule_map = {
        test_configs.TFLITE_MODEL.id:
            model_rule_generator.ModelRule(target_name=f"model-x",
                                           file_path="x.tflite",
                                           cmake_rules=["abc"]),
        test_configs.TF_MODEL.id:
            model_rule_generator.ModelRule(target_name=f"model-y",
                                           file_path="y_saved_model",
                                           cmake_rules=["abc"])
    }

    cmake_rules = iree_rule_generator.generate_rules(
        package_name="${package}",
        root_path=pathlib.PurePath("iree_root"),
        artifacts_root=artifacts_root,
        model_rule_map=model_rule_map)

    concated_cmake_rules = "\n".join(cmake_rules)
    self.assertRegex(concated_cmake_rules,
                     f"iree-imported-model-{test_configs.TFLITE_MODEL.id}")
    self.assertRegex(concated_cmake_rules,
                     f"iree-imported-model-{test_configs.TF_MODEL.id}")
    self.assertRegex(
        concated_cmake_rules,
        f"iree-module-{test_configs.TFLITE_MODEL.id}-{test_configs.COMPILE_CONFIG_A.id}"
    )
    self.assertRegex(
        concated_cmake_rules,
        f"iree-module-{test_configs.TFLITE_MODEL.id}-{test_configs.COMPILE_CONFIG_B.id}"
    )
    self.assertRegex(
        concated_cmake_rules,
        f"iree-module-{test_configs.TF_MODEL.id}-{test_configs.COMPILE_CONFIG_A.id}"
    )


if __name__ == "__main__":
  unittest.main()
