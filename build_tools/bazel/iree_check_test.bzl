# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that run a module using iree-check-module."""

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:native_binary.bzl", "native_test")

DEFAULT_TARGET_BACKENDS_AND_DRIVERS = [
    ("vmvx", "local-task"),
    ("vulkan-spirv", "vulkan"),
    ("llvm-cpu", "local-task"),
]

def iree_check_test(
        name,
        src,
        target_backend,
        driver = None,
        compiler_flags = [],
        input_type = None,
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        **kwargs):
    """Creates an iree-check-module test for the specified source file.

    Args:
      name: name of the generated test.
      src: source mlir file containing the module.
      target_backend: target backend to compile for.
      driver: driver to run the module with. This can be omitted to test only
          compilation, but consider omiting the driver as a hacky abuse of the
          rule since compilation on its own not use iree-check-module.
      compiler_flags: additional flags to pass to the compiler. Bytecode output
          format and backend flags are passed automatically.
      input_type: Value to pass to --iree-input-type.
      runner_args: additional runner_args to pass to iree-check-module. The
          driver and input file are passed automatically.
      tags: additional tags to apply to the generated test. Tag
          "driver=DRIVER" and "target=TARGET" are added automatically.
      timeout: timeout for the generated tests.
      **kwargs: any additional attributes to pass to the underlying native_test.
    """

    input_type_flags = []
    if input_type:
        input_type_flags = ["--iree-input-type=%s" % input_type]
    flags = [
        "--iree-hal-target-backends=%s" % target_backend,
    ] + compiler_flags + input_type_flags
    bytecode_module_name = name + "_bytecode_module"

    iree_bytecode_module(
        name = bytecode_module_name,
        src = src,
        flags = flags,
        tags = ["target=%s" % target_backend],
        deps = deps,
        visibility = ["//visibility:private"],
    )

    if not driver:
        return

    native_test(
        name = name,
        args = [
            "--device=%s" % driver,
            "--module=$(location :%s)" % bytecode_module_name,
        ] + runner_args,
        data = [":%s" % bytecode_module_name],
        src = "//tools:iree-check-module",
        tags = tags + ["driver=%s" % driver, "target=%s" % target_backend],
        timeout = timeout,
        **kwargs
    )

def iree_check_single_backend_test_suite(
        name,
        srcs,
        target_backend,
        driver = None,
        compiler_flags = [],
        input_type = None,
        runner_args = [],
        tags = [],
        deps = [],
        timeout = None,
        **kwargs):
    """Creates a test suite of iree-check-module tests for a single backend/driver pair.

    One test is generated per source file.

    Args:
      name: name of the generated test suite.
      srcs: source mlir files containing the module.
      target_backend: target backend to compile for.
      driver: driver to run the module with. This can be omitted to test only
          compilation, but consider omiting the driver as a hacky abuse of the
          rule since compilation on its own not use iree-check-module.
      compiler_flags: additional flags to pass to the compiler. Bytecode output
          format and backend flags are passed automatically.
      input_type: Value to pass to --iree-input-type.
      runner_args: additional runner_args to pass to the underlying
          iree-check-module tests. The driver and input file are passed
          automatically. To use different runner_args per test, create a
          separate suite or iree_check_test.
      tags: tags to apply to the generated tests. Note that as in standard test
          suites, manual is treated specially and will also apply to the test
          suite itself.
      timeout: timeout for the generated tests.
      **kwargs: any additional attributes to pass to the underlying tests and
          test suite.
    """

    # Metal backend/driver not supported by Bazel build.
    if target_backend == "metal-spirv" or driver == "metal":
        return

    # ROCm/HIP backend/driver not supported by Bazel build.
    if target_backend == "rocm" or driver == "hip":
        return

    tests = []
    for src in srcs:
        test_name = "_".join([name, src]).replace("/", "_").replace(":", "_")
        iree_check_test(
            name = test_name,
            src = src,
            target_backend = target_backend,
            driver = driver,
            compiler_flags = compiler_flags,
            input_type = input_type,
            runner_args = runner_args,
            tags = tags,
            timeout = timeout,
            deps = deps,
            **kwargs
        )
        tests.append(test_name)

    if not driver:
        return

    native.test_suite(
        name = name,
        tests = tests,
        tags = tags + ["driver=%s" % driver, "target=%s" % target_backend],
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )

def iree_check_test_suite(
        name,
        srcs,
        target_backends_and_drivers = DEFAULT_TARGET_BACKENDS_AND_DRIVERS,
        compiler_flags = [],
        input_type = None,
        runner_args = [],
        tags = [],
        target_cpu_features_variants = [],
        deps = [],
        **kwargs):
    """Creates a test suite of iree-check-module tests.

    One test is generated per source file and backend/driver.

    Args:
      name: name of the generated test suite.
      srcs: source mlir files containing the module.
      target_backends_and_drivers: backend/driver pairs to compile and run the
          module, respectively.
      compiler_flags: additional flags to pass to the compiler. Bytecode output
          format and backend flags are passed automatically.
      input_type: Value to pass to --iree-input-type.
      runner_args: additional runner_args to pass to the underlying
          iree-check-module tests. The driver and input file are passed
          automatically. To use different runner_args per test, create a
          separate suite or iree_check_test.
      tags: tags to apply to the generated tests. Note that as in standard test
          suites, manual is treated specially and will also apply to the test
          suite itself.
      target_cpu_features_variants: ignored, assumed to be ["generic"] in this
          Bazel implementation. See the CMake implementation for what this does
          in general.
      **kwargs: any additional attributes to pass to the underlying tests and
          test suite.
    """

    # Like CMake, default to "generic". Unlike CMake, do not honor other values.
    generic_flags = compiler_flags + ["--iree-llvmcpu-target-cpu=generic"]

    # We could have complicated argument override logic for runner_args and such, or... the client
    # could just create a test suite. The latter seems simpler and more readable.
    tests = []
    for backend, driver in target_backends_and_drivers:
        suite_name = "_".join([name, backend, driver])
        iree_check_single_backend_test_suite(
            name = suite_name,
            srcs = srcs,
            driver = driver,
            target_backend = backend,
            compiler_flags = generic_flags,
            input_type = input_type,
            runner_args = runner_args,
            tags = tags,
            deps = deps,
            **kwargs
        )
        tests.append(suite_name)
    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )
