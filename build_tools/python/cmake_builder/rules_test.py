#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap
import unittest
import cmake_builder.rules


class RulesTest(unittest.TestCase):

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

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_bytecode_module(
          NAME
            "abcd"
          SRC
            "abcd.mlir"
          MODULE_FILE_NAME
            "abcd.vmfb"
          C_IDENTIFIER
            "abcd.c"
          COMPILE_TOOL
            "iree_iree-compile2"
          STATIC_LIB_PATH
            "libx.a"
          FLAGS
            "--backend=cpu"
            "--opt=3"
          DEPS
            "iree_libx"
            "iree_liby"
          TESTONLY
        )
        """))

  def test_build_iree_bytecode_module_with_defaults(self):
    rule = cmake_builder.rules.build_iree_bytecode_module(
        target_name="abcd",
        src="abcd.mlir",
        module_name="abcd.vmfb",
        flags=["--backend=cpu", "--opt=3"])

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_bytecode_module(
          NAME
            "abcd"
          SRC
            "abcd.mlir"
          MODULE_FILE_NAME
            "abcd.vmfb"
          FLAGS
            "--backend=cpu"
            "--opt=3"
          PUBLIC
        )
        """))

  def test_build_iree_fetch_artifact(self):
    rule = cmake_builder.rules.build_iree_fetch_artifact(
        target_name="abcd",
        source_url="https://example.com/abcd.tflite",
        output="./abcd.tflite",
        unpack=True)

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_fetch_artifact(
          NAME
            "abcd"
          SOURCE_URL
            "https://example.com/abcd.tflite"
          OUTPUT
            "./abcd.tflite"
          UNPACK
        )
        """))

  def test_build_iree_import_tf_model(self):
    rule = cmake_builder.rules.build_iree_import_tf_model(
        target_path="pkg_abcd",
        source="abcd/model",
        entry_function="main",
        output_mlir_file="abcd.mlir")

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_import_tf_model(
          TARGET_NAME
            "pkg_abcd"
          SOURCE
            "abcd/model"
          ENTRY_FUNCTION
            "main"
          OUTPUT_MLIR_FILE
            "abcd.mlir"
        )
        """))

  def test_build_iree_import_tflite_model(self):
    rule = cmake_builder.rules.build_iree_import_tflite_model(
        target_path="pkg_abcd",
        source="abcd.tflite",
        output_mlir_file="abcd.mlir")

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_import_tflite_model(
          TARGET_NAME
            "pkg_abcd"
          SOURCE
            "abcd.tflite"
          OUTPUT_MLIR_FILE
            "abcd.mlir"
        )
        """))

  def test_build_add_dependencies(self):
    rule = cmake_builder.rules.build_add_dependencies(
        target="iree_mlir_suites", deps=["pkg_abcd", "pkg_efgh"])

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        add_dependencies(iree_mlir_suites
          pkg_abcd
          pkg_efgh
        )
        """))


if __name__ == "__main__":
  unittest.main()
