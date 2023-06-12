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
          NAME "abcd"
          SRC "abcd.mlir"
          MODULE_FILE_NAME "abcd.vmfb"
          C_IDENTIFIER "abcd.c"
          COMPILE_TOOL "iree_iree-compile2"
          STATIC_LIB_PATH "libx.a"
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
          NAME "abcd"
          SRC "abcd.mlir"
          MODULE_FILE_NAME "abcd.vmfb"
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
          NAME "abcd"
          SOURCE_URL "https://example.com/abcd.tflite"
          OUTPUT "./abcd.tflite"
          UNPACK
        )
        """))

  def test_build_iree_import_tf_model(self):
    rule = cmake_builder.rules.build_iree_import_tf_model(
        target_path="pkg_abcd",
        source="abcd/model",
        import_flags=[
            "--tf-savedmodel-exported-names=main",
            "--tf-import-type=savedmodel_v1"
        ],
        output_mlir_file="abcd.mlir")

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_import_tf_model(
          TARGET_NAME "pkg_abcd"
          SOURCE "abcd/model"
          IMPORT_FLAGS
            "--tf-savedmodel-exported-names=main"
            "--tf-import-type=savedmodel_v1"
          OUTPUT_MLIR_FILE "abcd.mlir"
        )
        """))

  def test_build_iree_import_tflite_model(self):
    rule = cmake_builder.rules.build_iree_import_tflite_model(
        target_path="pkg_abcd",
        source="abcd.tflite",
        import_flags=["--fake-flag=abcd"],
        output_mlir_file="abcd.mlir")

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_import_tflite_model(
          TARGET_NAME "pkg_abcd"
          SOURCE "abcd.tflite"
          IMPORT_FLAGS
            "--fake-flag=abcd"
          OUTPUT_MLIR_FILE "abcd.mlir"
        )
        """))

  def test_build_iree_benchmark_suite_module_test(self):
    rule = cmake_builder.rules.build_iree_benchmark_suite_module_test(
        target_name="model_test",
        driver="LOCAL_TASK",
        expected_output="xyz",
        platform_module_map={
            "x86_64": "a.vmfb",
            "arm": "b.vmfb"
        },
        runner_args=["--x=0", "--y=1"],
        timeout_secs=10,
        labels=["defaults", "e2e"],
        xfail_platforms=["arm_64-Android", "riscv_32-Linux"])

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        iree_benchmark_suite_module_test(
          NAME "model_test"
          DRIVER "LOCAL_TASK"
          EXPECTED_OUTPUT "xyz"
          TIMEOUT "10"
          MODULES
            "x86_64=a.vmfb"
            "arm=b.vmfb"
          RUNNER_ARGS
            "--x=0"
            "--y=1"
          LABELS
            "defaults"
            "e2e"
          XFAIL_PLATFORMS
            "arm_64-Android"
            "riscv_32-Linux"
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

  def test_build_set(self):
    rule = cmake_builder.rules.build_set(variable_name="_ABC", value="123")

    self.assertEqual(
        rule,
        textwrap.dedent("""\
        set(_ABC
          123
        )
        """))


if __name__ == "__main__":
  unittest.main()
