# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Creates a binary and a test for a cc benchmark target.

It's good to test that benchmarks run, but it's really annoying to run a billion
iterations of them every time you try to run tests. So we create these as
binaries and then invoke them as tests with `--benchmark_min_time=0`.
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
        **kwargs):
    """Creates a binary and a test for a cc benchmark target.

    Arguments passed to the binary target:
      name, srcs, data, deps, copts, defines, linkopts, tags, testonly, **kwargs
    Arguments passed to the test target:
      {name}_test, tags, size, timeout, **kwargs
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

    native_test(
        name = "{}_test".format(name),
        src = name,
        size = size,
        tags = tags,
        timeout = timeout,
        args = ["--benchmark_min_time=0"],
        **kwargs
    )
