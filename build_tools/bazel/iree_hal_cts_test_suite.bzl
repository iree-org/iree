# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates CTS test binaries for a HAL driver.

Usage in a driver's cts/BUILD.bazel:

    load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")
    load("//build_tools/bazel:iree_hal_cts_test_suite.bzl", "iree_hal_cts_test_suite")

    iree_runtime_cc_library(
        name = "backends",
        testonly = True,
        srcs = ["backends.cc"],
        deps = [...],
        alwayslink = True,
    )

    iree_hal_cts_test_suite(
        backends_lib = ":backends",
        executable_formats = {
            "vmvx": {
                "target_device": "local",
                "flags": ["--iree-hal-local-target-device-backends=vmvx"],
                "identifier": "iree_cts_testdata_vmvx",
                "backend_name": "local_task",
                "format_string": '"vmvx-bytecode-fb"',
            },
        },
        testdata = "//runtime/src/iree/hal/cts/testdata:executable_srcs",
    )
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_hal_executable.bzl", "iree_hal_executables")

# Non-executable test categories. Each entry maps a test binary name suffix
# to the aggregate test library it links.
_NON_EXECUTABLE_SUITES = [
    ("buffer_tests", "//runtime/src/iree/hal/cts/buffer:all_tests"),
    ("command_buffer_tests", "//runtime/src/iree/hal/cts/command_buffer:all_tests"),
    ("core_tests", "//runtime/src/iree/hal/cts/core:all_tests"),
    ("file_tests", "//runtime/src/iree/hal/cts/file:all_tests"),
    ("queue_tests", "//runtime/src/iree/hal/cts/queue:all_tests"),
]

# Executable-dependent test categories. Each entry maps a test binary name
# suffix to the aggregate test library it links.
_EXECUTABLE_SUITES = [
    ("dispatch_tests", "//runtime/src/iree/hal/cts/command_buffer:all_dispatch_tests"),
    ("executable_tests", "//runtime/src/iree/hal/cts/core:all_executable_tests"),
]

def _camel_case(snake_str):
    """Converts snake_case to CamelCase: 'llvm_cpu' -> 'LlvmCpu'."""
    result = ""
    for part in snake_str.split("_"):
        result += part.capitalize()
    return result

def _cts_testdata_gen_impl(ctx):
    """Expands the testdata_format.cc.tpl template with build setting resolution.

    Like expand_template, but resolves template variables in substitution
    values from build settings specified in flag_values. Non-build-setting
    entries (file targets) are ignored — they only apply to compiler flag
    resolution in iree_hal_executables, not to C++ template expansion.
    """
    substitutions = dict(ctx.attr.substitutions)
    for target, placeholder in ctx.attr.flag_values.items():
        if BuildSettingInfo not in target:
            continue
        value = target[BuildSettingInfo].value
        template = "{%s}" % placeholder
        substitutions = {
            key: val.replace(template, value)
            for key, val in substitutions.items()
        }

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = substitutions,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_cts_testdata_gen = rule(
    implementation = _cts_testdata_gen_impl,
    attrs = {
        "template": attr.label(mandatory = True, allow_single_file = True),
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(),
        "flag_values": attr.label_keyed_string_dict(
            allow_files = True,
        ),
    },
)

def iree_hal_cts_testdata(
        format_name,
        target_device,
        identifier,
        backend_name,
        format_string,
        testdata,
        flags = [],
        flag_values = {},
        data = [],
        testonly = True,
        **kwargs):
    """Compiles CTS test executables and creates a testdata registration library.

    Use this directly when multiple iree_hal_cts_test_suite() calls need to
    share the same compiled executables (e.g., CUDA graph/stream variants).
    For single-variant drivers, use executable_formats in iree_hal_cts_test_suite
    instead -- it calls this internally.

    Returns the label of the generated testdata library (e.g., ":testdata_cuda_lib").

    Args:
        format_name: Short name (e.g., "vmvx", "cuda", "hip").
        target_device: Target device for iree-compile.
        identifier: C identifier for the embedded data.
        backend_name: Backend name for CtsRegistry registration.
        format_string: C expression for the format string. May contain
            {PLACEHOLDER} template variables resolved from flag_values.
        testdata: Filegroup label for MLIR test sources (e.g.,
            "//runtime/src/iree/hal/cts/testdata:executable_srcs").
        flags: Compiler flags. May contain {PLACEHOLDER} template variables
            resolved from flag_values.
        flag_values: Dict mapping placeholder names to target labels.
            See iree_hal_executable() for details.
        data: Additional files for the compile action inputs.
        testonly: Defaults to True.
        **kwargs: Forwarded to underlying rules.
    """
    testdata_name = "testdata_%s" % format_name

    # iree_hal_executables() is a macro that inverts flag_values internally,
    # so pass the user-facing form directly.
    iree_hal_executables(
        name = testdata_name,
        srcs = [testdata],
        target_device = target_device,
        flags = flags,
        flag_values = flag_values,
        data = data,
        identifier = identifier,
        testonly = testonly,
        **kwargs
    )

    gen_cc_name = "%s_gen" % testdata_name
    gen_cc_file = "%s.cc" % testdata_name
    header_path = "%s/%s.h" % (native.package_name(), testdata_name)
    func_name = _camel_case(format_name)

    # Invert to {"//label": "PLACEHOLDER"} for label_keyed_string_dict.
    # File targets pass through to the rule but are ignored during template
    # expansion (only BuildSettingInfo entries apply to format_string).
    rule_flag_values = {v: k for k, v in flag_values.items()}
    _cts_testdata_gen(
        name = gen_cc_name,
        template = "//runtime/src/iree/hal/cts/util:testdata_format.cc.tpl",
        out = gen_cc_file,
        substitutions = {
            "{HEADER_PATH}": header_path,
            "{FORMAT_FUNC_NAME}": func_name,
            "{IDENTIFIER}": identifier,
            "{FORMAT_VAR_NAME}": "%s_format" % format_name,
            "{BACKEND_NAME}": backend_name,
            "{FORMAT_NAME}": format_name,
            "{FORMAT_STRING}": format_string,
        },
        flag_values = rule_flag_values,
        testonly = testonly,
    )

    testdata_lib_name = "%s_lib" % testdata_name
    iree_runtime_cc_library(
        name = testdata_lib_name,
        testonly = testonly,
        srcs = [gen_cc_file],
        deps = [
            ":%s" % testdata_name,
            "//runtime/src/iree/hal/cts/util:registry",
        ],
        alwayslink = True,
    )

    return ":%s" % testdata_lib_name

def iree_hal_cts_test_suite(
        backends_lib,
        executable_formats = {},
        testdata_libs = [],
        testdata = None,
        flag_values = {},
        name = "",
        args = [],
        resource_group = None,
        tags = [],
        testonly = True,
        **kwargs):
    """Generates CTS test binaries for a HAL driver.

    Creates non-executable test binaries (core, buffer, command_buffer, queue,
    file) that link against the provided backends library. If executable_formats
    is provided, also compiles MLIR test sources for each format and creates
    executable and dispatch test binaries.

    Args:
        backends_lib: Label of the hand-written backends.cc library that
            registers the driver with CtsRegistry.
        executable_formats: Dict mapping format names to config dicts. Each
            config dict has keys:
              target_device: Target device for iree-compile (e.g., "local").
              flags: List of compiler flags.
              identifier: C identifier for the embedded data (used to derive
                  the _create() function name in the generated header).
              backend_name: Backend name string for CtsRegistry registration.
              format_string: C expression for the executable format string
                  (e.g., '"vmvx-bytecode-fb"' or '"embedded-elf-" IREE_ARCH').
            Mutually exclusive with testdata_libs.
        testdata_libs: Pre-built testdata library labels for multi-variant
            drivers. When multiple iree_hal_cts_test_suite() calls share
            the same compiled executables (e.g., CUDA graph/stream variants),
            define the testdata targets once and pass them here instead of
            using executable_formats. Mutually exclusive with executable_formats.
        testdata: Filegroup label for MLIR test sources (e.g.,
            "//runtime/src/iree/hal/cts/testdata:executable_srcs").
            Required when executable_formats is provided.
        flag_values: Dict mapping string_flag build setting labels to
            placeholder names. Forwarded to iree_hal_cts_testdata when
            using executable_formats.
        name: Optional name prefix for generated targets. When empty, targets
            are named directly (core_tests, buffer_tests, etc.). When set,
            targets are prefixed (stream_core_tests, graph_buffer_tests, etc.).
            Use a prefix for multi-variant drivers (e.g., CUDA graph/stream).
        args: Runtime arguments passed to all test binaries.
        resource_group: Optional shared resource group for generated tests.
            Tests sharing the same resource group will not run concurrently.
        tags: Additional tags for test targets.
        testonly: Defaults to True.
        **kwargs: Forwarded to underlying rules (e.g., target_compatible_with).
    """

    # Separate test-specific kwargs (env, data, size, etc.) from kwargs that
    # apply to all targets (target_compatible_with, etc.). Test kwargs go only
    # to iree_runtime_cc_test; the rest go to all generated targets.
    test_kwargs = {}
    for key in ("env", "env_inherit", "data", "size", "timeout", "flaky", "shard_count", "local"):
        if key in kwargs:
            test_kwargs[key] = kwargs.pop(key)

    # Build the name prefix: "name_" if set, "" otherwise.
    prefix = ("%s_" % name) if name else ""

    if executable_formats and not testdata:
        fail("iree_hal_cts_test_suite: testdata is required when executable_formats is provided")

    # Use pre-built testdata libs if provided, otherwise compile from formats.
    _testdata_libs = list(testdata_libs)
    for format_name, config in executable_formats.items():
        lib_label = iree_hal_cts_testdata(
            format_name = format_name,
            target_device = config["target_device"],
            identifier = config["identifier"],
            backend_name = config["backend_name"],
            format_string = config["format_string"],
            testdata = testdata,
            flags = config.get("flags", []),
            flag_values = flag_values,
            testonly = testonly,
            **kwargs
        )
        _testdata_libs.append(lib_label)

    # Common deps for all test binaries.
    common_deps = [
        backends_lib,
        "//runtime/src/iree/hal/cts/util:registry",
        "//runtime/src/iree/hal/cts/util:test_base",
        "//runtime/src/iree/testing:gtest",
    ]

    # Merge test-specific and general kwargs for test targets.
    all_test_kwargs = dict(kwargs)
    all_test_kwargs.update(test_kwargs)

    # Non-executable test binaries.
    for suffix, test_lib in _NON_EXECUTABLE_SUITES:
        iree_runtime_cc_test(
            name = "%s%s" % (prefix, suffix),
            srcs = ["//runtime/src/iree/hal/cts/util:test_main.cc"],
            args = args,
            deps = common_deps + [test_lib],
            resource_group = resource_group,
            tags = tags,
            **all_test_kwargs
        )

    # Executable-dependent test binaries (only if formats are configured).
    if _testdata_libs:
        for suffix, test_lib in _EXECUTABLE_SUITES:
            iree_runtime_cc_test(
                name = "%s%s" % (prefix, suffix),
                srcs = ["//runtime/src/iree/hal/cts/util:test_main.cc"],
                args = args,
                deps = common_deps + _testdata_libs + [test_lib],
                resource_group = resource_group,
                tags = tags,
                **all_test_kwargs
            )
