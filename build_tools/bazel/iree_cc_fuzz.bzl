# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining libFuzzer-based fuzz targets.

Fuzz targets require --config=fuzzer to build properly. The config instruments
all code for coverage feedback and adds appropriate compile/link flags.

Example usage:
    load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_fuzz")

    iree_runtime_cc_fuzz(
        name = "unicode_fuzz",
        srcs = ["unicode_fuzz.cc"],
        deps = [":unicode"],
    )

Building and running:
    bazel build --config=fuzzer //path/to:unicode_fuzz
    ./bazel-bin/path/to/unicode_fuzz corpus/ -max_total_time=60
"""

def iree_cc_fuzz(
        name,
        srcs,
        deps = None,
        data = None,
        copts = None,
        defines = None,
        linkopts = None,
        tags = None,
        **kwargs):
    """Creates a libFuzzer-based fuzz target.

    Args:
        name: Target name (e.g., "unicode_fuzz").
        srcs: Source files containing LLVMFuzzerTestOneInput().
        deps: Library dependencies.
        data: Data file dependencies.
        copts: Additional compile options.
        defines: Preprocessor definitions.
        linkopts: Additional link options.
        tags: Target tags. "fuzz" tag is added automatically.
        **kwargs: Additional cc_binary attributes.
    """
    if deps == None:
        deps = []
    if data == None:
        data = []
    if copts == None:
        copts = []
    if defines == None:
        defines = []
    if linkopts == None:
        linkopts = []
    if tags == None:
        tags = []

    # Add "fuzz" tag if not present.
    if "fuzz" not in tags:
        tags = tags + ["fuzz"]

    native.cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        copts = copts,
        defines = defines + ["FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"],
        linkopts = linkopts + ["-fsanitize=fuzzer"],
        tags = tags,
        testonly = True,
        **kwargs
    )

def iree_runtime_cc_fuzz(deps = None, **kwargs):
    """Fuzz target for runtime code using libFuzzer.

    Wraps iree_cc_fuzz and adds //runtime/src:runtime_defines dependency.

    Args:
        deps: Library dependencies (runtime_defines added automatically).
        **kwargs: Additional arguments passed to iree_cc_fuzz.
    """
    if deps == None:
        deps = []
    iree_cc_fuzz(
        deps = deps + ["//runtime/src:runtime_defines"],
        **kwargs
    )

def iree_compiler_cc_fuzz(deps = None, **kwargs):
    """Fuzz target for compiler code using libFuzzer.

    Wraps iree_cc_fuzz and adds //compiler/src:defs dependency.

    Args:
        deps: Library dependencies (compiler defs added automatically).
        **kwargs: Additional arguments passed to iree_cc_fuzz.
    """
    if deps == None:
        deps = []
    iree_cc_fuzz(
        deps = deps + ["//compiler/src:defs"],
        **kwargs
    )
