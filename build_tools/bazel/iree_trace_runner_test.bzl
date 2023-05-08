# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that use a trace-runner."""

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:native_binary.bzl", "native_test")

def iree_trace_runner_test(
        name,
        src,
        module_name,
        target_backend,
        driver,
        trace_runner,
        trace,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        target_cpu_features = None,
        timeout = None,
        **kwargs):
    """Creates a test running a custom trace-runner on a trace file (yaml).

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the trace-runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        trace_runner: trace-runner program to run.
        trace: trace file input to the trace-runner program.
        module_name: specifies the  path to use for the enerated IREE module
            (.vmfb). Mandatory, unlike in iree_check_test, because trace files
            (.yaml) reference a specific module file path.
        timeout: timeout for the generated tests.
        target_cpu_features: currently unimplemented (must be empty), will
            eventually allow specifying target CPU features.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    if target_cpu_features:
        fail("target_cpu_features must currently be empty")

    bytecode_module_name = name + "_bytecode_module"
    iree_bytecode_module(
        name = bytecode_module_name,
        module_name = module_name,
        src = src,
        flags = [
            "--mlir-print-op-on-diagnostic=false",
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
            "$(location :%s)" % trace,
        ] + runner_args,
        data = [
            ":%s" % bytecode_module_name,
            ":%s" % trace,
        ],
        src = trace_runner,
        tags = tags + ["driver=%s" % driver],
        timeout = timeout,
        **kwargs
    )

def iree_single_backend_generated_trace_runner_test(
        name,
        generator,
        trace_runner,
        target_backend,
        driver,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        target_cpu_features = None,
        timeout = None,
        **kwargs):
    """Generates an iree_trace_runner_test using a custom python generator script.

    The generator script produces a .mlir source and a .yaml trace, which are
    passed to iree_trace_runner_test.

    Args:
        name: Name of the target
        generator: Target to run to generate the source file and trace files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_code=(current binary dir)/name.mlir
            --output_trace=(current binary dir)/name.yaml
            --module_path=(current binary dir)/name.vmfb
        generator_args: additional args to pass to the generator program.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the trace-runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        trace_runner: trace-runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features: currently unimplemented (must be empty), will
            eventually allow specifying target CPU features.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    if target_cpu_features:
        fail("target_cpu_features must currently be empty")

    src = "%s.mlir" % (name)
    trace = "%s.yaml" % (name)
    module_name = "%s.vmfb" % (name)
    native.genrule(
        name = "%s_generate" % (name),
        outs = [src, trace],
        cmd = " ".join([
            "$(location %s)" % (generator),
            " ".join([('"%s"' % arg) for arg in generator_args]),
            "--output_code=$(location %s)" % (src),
            "--output_trace=$(location %s)" % (trace),
            # Explanation for why "$(RULEDIR)/%s" instead of "$(location %s)" below:
            # module_path points to a file that does not yet exist as it will
            # be generated by iree_bytecode_module below iree_trace_runner_test.
            "--module_path=$(RULEDIR)/%s" % (module_name),
        ] + [('"%s"' % arg) for arg in generator_args]),
        tools = [generator],
        message = "Generating code and trace for test %s..." % (name),
        output_to_bindir = 1,
        testonly = True,
        **kwargs
    )
    iree_trace_runner_test(
        name = name,
        src = src,
        module_name = module_name,
        target_backend = target_backend,
        driver = driver,
        trace_runner = trace_runner,
        trace = trace,
        compiler_flags = compiler_flags,
        runner_args = runner_args,
        tags = tags,
        timeout = timeout,
        **kwargs
    )

def iree_generated_trace_runner_test(
        name,
        generator,
        trace_runner,
        target_backends_and_drivers,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        target_cpu_features_variants = [],
        **kwargs):
    """Generates a suite of iree_trace_runner_test on multiple backends/drivers.

    Args:
        name: Name of the target
        generator: Target to run to generate the source file and trace files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_code=(current binary dir)/name.mlir
            --output_trace=(current binary dir)/name.yaml
            --module_path=(current binary dir)/name.vmfb
        generator_args: additional args to pass to the generator program.
        target_backends_and_drivers: backend/driver pairs to compile and run
            the module.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the trace-runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        trace_runner: trace-runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features_variants: list of target cpu features variants.
            Currently unimplemented, so each entry must be either "default" or
            start with "arm_64:" so as Bazel builds are currently x86-only,
            we know that it is correct to ignore this.
        **kwargs: any additional attributes to pass to the underlying tests and test suite.
    """

    for target_cpu_features in target_cpu_features_variants:
        if not (target_cpu_features == "default" or target_cpu_features.startswith("arm_64:")):
            fail("Entry %s in target_cpu_features_variants: unimplemented" % target_cpu_features)

    tests = []
    for backend, driver in target_backends_and_drivers:
        # CUDA backend/driver not supported by Bazel build.
        if backend == "cuda" or driver == "cuda":
            continue
        suite_entry_name = "_".join([name, backend, driver])
        iree_single_backend_generated_trace_runner_test(
            name = suite_entry_name,
            generator = generator,
            trace_runner = trace_runner,
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
