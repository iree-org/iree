#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import cmake_builder.rules


class RulesTest(unittest.TestCase):

  def test_get_target_path(self):
    self.assertEqual(cmake_builder.rules.get_target_path("abcd"),
                     "${_PACKAGE_NAME}_abcd")

  def test_build_iree_bytecode_module(self):
    rule = cmake_builder.rules.build_iree_bytecode_module(
        target_name="abcd",
        src="abcd.mlir",
        module_name="abcd.vmfb",
        flags=["--backend=cpu", "--opt=3"],
        compile_tool_target="iree_iree-compile2",
        c_identifier="abcd.c",
        static_lib_path="libx.a",
        deps=["iree_libx", "iree_liby"],
        testonly=True,
        public=False)

    self.assertEqual(rule, ('iree_bytecode_module(\n'
                            '  NAME\n'
                            '    "abcd"\n'
                            '  SRC\n'
                            '    "abcd.mlir"\n'
                            '  MODULE_FILE_NAME\n'
                            '    "abcd.vmfb"\n'
                            '  C_IDENTIFIER\n'
                            '    "abcd.c"\n'
                            '  COMPILE_TOOL\n'
                            '    "iree_iree-compile2"\n'
                            '  STATIC_LIB_PATH\n'
                            '    "libx.a"\n'
                            '  FLAGS\n'
                            '    "--backend=cpu"\n'
                            '    "--opt=3"\n'
                            '  DEPS\n'
                            '    "iree_libx"\n'
                            '    "iree_liby"\n'
                            '  TESTONLY\n'
                            ')\n'))

  def test_build_iree_bytecode_module_with_defaults(self):
    rule = cmake_builder.rules.build_iree_bytecode_module(
        target_name="abcd",
        src="abcd.mlir",
        module_name="abcd.vmfb",
        flags=["--backend=cpu", "--opt=3"])

    self.assertEqual(rule, ('iree_bytecode_module(\n'
                            '  NAME\n'
                            '    "abcd"\n'
                            '  SRC\n'
                            '    "abcd.mlir"\n'
                            '  MODULE_FILE_NAME\n'
                            '    "abcd.vmfb"\n'
                            '  FLAGS\n'
                            '    "--backend=cpu"\n'
                            '    "--opt=3"\n'
                            '  PUBLIC\n'
                            ')\n'))

  def test_build_iree_fetch_artifact(self):
    rule = cmake_builder.rules.build_iree_fetch_artifact(
        target_name="abcd",
        source_url="https://example.com/abcd.tflite",
        output="./abcd.tflite",
        unpack=True)

    self.assertEqual(rule, ('iree_fetch_artifact(\n'
                            '  NAME\n'
                            '    "abcd"\n'
                            '  SOURCE_URL\n'
                            '    "https://example.com/abcd.tflite"\n'
                            '  OUTPUT\n'
                            '    "./abcd.tflite"\n'
                            '  UNPACK\n'
                            ')\n'))

  def test_build_iree_import_tf_model(self):
    rule = cmake_builder.rules.build_iree_import_tf_model(
        target_path="pkg_abcd",
        source="abcd/model",
        entry_function="main",
        output_mlir_file="abcd.mlir")

    self.assertEqual(rule, ('iree_import_tf_model(\n'
                            '  TARGET_NAME\n'
                            '    "pkg_abcd"\n'
                            '  SOURCE\n'
                            '    "abcd/model"\n'
                            '  ENTRY_FUNCTION\n'
                            '    "main"\n'
                            '  OUTPUT_MLIR_FILE\n'
                            '    "abcd.mlir"\n'
                            ')\n'))

  def test_build_iree_import_tflite_model(self):
    rule = cmake_builder.rules.build_iree_import_tflite_model(
        target_path="pkg_abcd",
        source="abcd.tflite",
        output_mlir_file="abcd.mlir")

    self.assertEqual(rule, ('iree_import_tflite_model(\n'
                            '  TARGET_NAME\n'
                            '    "pkg_abcd"\n'
                            '  SOURCE\n'
                            '    "abcd.tflite"\n'
                            '  OUTPUT_MLIR_FILE\n'
                            '    "abcd.mlir"\n'
                            ')\n'))

  def test_build_add_dependencies(self):
    rule = cmake_builder.rules.build_add_dependencies(
        target="iree_mlir_suites", deps=["pkg_abcd", "pkg_efgh"])

    self.assertEqual(rule, ('add_dependencies(iree_mlir_suites\n'
                            '  pkg_abcd\n'
                            '  pkg_efgh\n'
                            ')\n'))


if __name__ == "__main__":
  unittest.main()
