# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Creates a binary and a test for a cc benchmark target.

It's good to test that benchmarks run, but it's really annoying to run a billion
iterations of them every time you try to run tests. So we create these as
binaries and then invoke them as tests with `--benchmark_min_time=0s`.
"""

load(":native_binary.bzl", "native_test")

def cc_binary_benchmark(
        name,
        srcs = None,
        data = None,
        deps = None,
        copts = None,
        defines = None,
        linkopts = None,
        tags = None,
        testonly = True,
        size = "small",
        timeout = None,
        args = None,
        **kwargs):
    """Creates a binary and a test for a cc benchmark target.

    Arguments passed to the binary target:
      name, srcs, data, deps, copts, defines, linkopts, tags, testonly, **kwargs
    Arguments passed to the test target:
      {name}_test, tags, size, timeout, args (merged with --benchmark_min_time=0s), **kwargs
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        data = data,
        deps = deps,
        copts = copts,
        defines = defines,
        linkopts = linkopts,
        tags = tags,
        testonly = testonly,
        **kwargs
    )

    # Merge args: enforced flag first, then user args.
    test_args = ["--benchmark_min_time=0s"]
    if args:
        test_args = test_args + args

    native_test(
        name = "{}_test".format(name),
        src = name,
        size = size,
        tags = tags,
        timeout = timeout,
        args = test_args,
        **kwargs
    )
