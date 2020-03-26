# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Macros for defining tests that run a module using iree-check-module."""

load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

ALL_TARGET_BACKENDS_AND_DRIVERS = [
    ("vmla", "vmla"),
    ("vulkan-spirv", "vulkan"),
    ("llvm-ir", "llvm"),
]

def iree_check_test(name, src, target_backend, driver, args = [], tags = [], **kwargs):
    """Creates an iree-check-module test for the specified source file.

    Args:
      name: name of the generated test.
      src: source mlir file containing the module.
      target_backend: target backend to compile for.
      driver: driver to run the module with.
      args: additional args to pass to iree-check-module. The driver and input file are passed
            automatically.
      tags: additional tags to apply to the generated test. A tag "driver=DRIVER" is added
            automatically.
      **kwargs: any additional attributes to pass to the underlying run_binary_test.
    """
    bytecode_module_name = name + "_bytecode_module"
    iree_bytecode_module(
        name = bytecode_module_name,
        src = src,
        flags = [
            "-iree-mlir-to-vm-bytecode-module",
            "--iree-hal-target-backends=%s" % target_backend,
        ],
        visibility = ["//visibility:private"],
    )

    run_binary_test(
        name = name,
        args = [
            "--driver=%s" % driver,
            "--input_file=$(location :%s)" % bytecode_module_name,
        ] + args,
        data = [":%s" % bytecode_module_name],
        test_binary = "//iree/modules/check:iree-check-module",
        tags = tags + ["driver=%s" % driver],
        **kwargs
    )

def iree_check_test_suite(
        name,
        srcs,
        target_backends_and_drivers = ALL_TARGET_BACKENDS_AND_DRIVERS,
        args = [],
        tags = [],
        **kwargs):
    """Creates a test suite of iree-check-module tests.

    One test is generated per source and driver.

    Args:
      name: name of the generated test suite.
      srcs: source mlir files containing the module.
      target_backends_and_drivers: backend, driver pairs to compile and run the module, respectively.
      args: additional args to pass to the underlying iree-check-module tests. The driver and
            input file are passed automatically. To use different args per test, create a
            separate suite or iree_check_test.
      tags: tags to apply to the generated tests. Note that as in standard test suites, manual
            is treated specially and will also apply to the test suite itself.
      **kwargs: any additional attributes to pass to the underlying tests and test suite.
    """

    # We could have complicated argument override logic for args and such, or... we could just
    # create a separate target. The latter seems simpler and more readable.
    tests = []
    for backend, driver in target_backends_and_drivers:
        for src in srcs:
            test_name = "_".join([name, src, backend, driver])
            iree_check_test(
                name = test_name,
                src = src,
                driver = driver,
                target_backend = backend,
                args = args,
                tags = tags,
                **kwargs
            )
            tests.append(test_name)
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
