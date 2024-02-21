# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that use the iree-e2e-matmul-test runner."""

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:native_binary.bzl", "native_test")

def iree_e2e_matmul_test(
        name,
        matmul_src,
        matmuls_vmfb,
        calls_src,
        calls_vmfb,
        target_backend,
        driver,
        test_runner,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        target_cpu_features = None,
        timeout = None,
        **kwargs):
    """Creates a test using a specified test runner program.

    Args:
        name: Name of the target
        matmul_src: mlir source file with matmuls to be compiled.
        matmuls_vmfb: specifies the path to use for the generated IREE module.
        calls_src: mlir source file with calls to be compiled.
        calls_vmfb: specifies the path to use for the generated IREE module.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features: target CPU features. Only for llvm-cpu backend.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    if target_cpu_features:
        fail("target_cpu_features must currently be empty")

    iree_bytecode_module(
        name = name + "_matmuls_module",
        module_name = matmuls_vmfb,
        src = matmul_src,
        flags = [
            "--iree-hal-target-backends=%s" % target_backend,
        ] + ([
            "--iree-llvmcpu-target-cpu-features=%s" % target_cpu_features,
        ] if target_cpu_features else []) + compiler_flags,
        visibility = ["//visibility:private"],
        testonly = True,
        **kwargs
    )

    iree_bytecode_module(
        name = name + "_calls_module",
        module_name = calls_vmfb,
        src = calls_src,
        flags = [
            "--iree-hal-target-backends=%s" % target_backend,
        ] + compiler_flags,
        visibility = ["//visibility:private"],
        testonly = True,
        **kwargs
    )

    native_test(
        name = name,
        args = [
            "--device=%s" % driver,
            "--module=$(location :%s)" % matmuls_vmfb,
            "--module=$(location :%s)" % calls_vmfb,
        ] + runner_args,
        data = [
            ":%s" % matmuls_vmfb,
            ":%s" % calls_vmfb,
        ],
        src = test_runner,
        tags = tags + ["driver=%s" % driver],
        timeout = timeout,
        **kwargs
    )

def iree_single_backend_e2e_matmul_test(
        name,
        generator,
        test_runner,
        target_backend,
        driver,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        target_cpu_features = None,
        timeout = None,
        **kwargs):
    """Generates an iree_e2e_matmul_test using a custom python generator script.

    The generator script produces .mlir sources which are compiled and passed to
    iree_e2e_matmul_test.

    Args:
        name: Name of the target
        generator: Target to run to generate the source MLIR files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_matmuls_mlir=(current binary dir)/name_matmuls.mlir
            --output_calls_mlir=(current binary dir)/name_calls.mlir
        generator_args: additional args to pass to the generator program.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features: target CPU features. Only for llvm-cpu backend.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    matmul_src = "%s.mlir" % (name)
    matmuls_vmfb = "%s.vmfb" % (name)
    calls_src = "%s_calls.mlir" % (name)
    calls_vmfb = "%s_calls.vmfb" % (name)
    native.genrule(
        name = "%s_generate" % (name),
        outs = [matmul_src, calls_src],
        cmd = " ".join([
            "$(location %s)" % (generator),
            " ".join([('"%s"' % arg) for arg in generator_args]),
            "--output_matmuls_mlir=$(location %s)" % (matmul_src),
            "--output_calls_mlir=$(location %s)" % (calls_src),
        ] + [('"%s"' % arg) for arg in generator_args]),
        tools = [generator],
        message = "Generating code and calls for test %s..." % (name),
        output_to_bindir = 1,
        testonly = True,
        **kwargs
    )
    iree_e2e_matmul_test(
        name = name,
        matmul_src = matmul_src,
        matmuls_vmfb = matmuls_vmfb,
        calls_src = calls_src,
        calls_vmfb = calls_vmfb,
        target_backend = target_backend,
        driver = driver,
        test_runner = test_runner,
        compiler_flags = compiler_flags,
        runner_args = runner_args,
        tags = tags,
        timeout = timeout,
        target_cpu_features = target_cpu_features,
        **kwargs
    )

def iree_generated_e2e_matmul_test(
        name,
        generator,
        test_runner,
        target_backends_and_drivers,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        target_cpu_features_variants = [],
        **kwargs):
    """Generates a suite of iree_e2e_matmul_test on multiple backends/drivers.

    Args:
        name: Name of the target
        generator: Target to run to generate the source MLIR files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_matmuls_mlir=(current binary dir)/name_matmuls.mlir
            --output_calls_mlir=(current binary dir)/name_calls.mlir
        generator_args: additional args to pass to the generator program.
        target_backends_and_drivers: backend/driver pairs to compile and run
            the module.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features_variants: list of target cpu features variants.
            Currently unimplemented in Bazel due to difficulty of specializing
            to target architecture in Bazel. The following describes the
            semantics that this should have if implemented. Each
            entry is either "default" for the architecture defaults, or a colon-
            separated triple "arch:name:cpu_features" where "arch" filters
            for a target CPU architecture (in IREE_ARCH format), "name" is a
            short name for the CPU features set (used to generate target names)
            and cpu_features is a comma-separated list of LLVM target attributes
            to enable. Example:
              x86_64:avx2_fma:+avx,+avx2,+fma
        **kwargs: any additional attributes to pass to the underlying tests and test suite.
    """

    tests = []
    for backend, driver in target_backends_and_drivers:
        # CUDA/ROCm backend/driver not supported by Bazel build.
        if backend == "cuda" or driver == "cuda" or backend == "rocm" or driver == "hip":
            continue
        suite_entry_name = "_".join([name, backend, driver])
        iree_single_backend_e2e_matmul_test(
            name = suite_entry_name,
            generator = generator,
            test_runner = test_runner,
            driver = driver,
            target_backend = backend,
            generator_args = generator_args,
            compiler_flags = compiler_flags,
            runner_args = runner_args,
            tags = tags,
            timeout = timeout,
            **kwargs
        )
        tests.append(suite_entry_name)
    native.test_suite(
        name = name,
        tests = tests,
        tags = tags,
        **kwargs
    )
