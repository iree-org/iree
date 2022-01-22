# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""native_binary() and native_test() rule implementations.

Rewritten from the Bazel Skylib version pending several fixes and improvements
to that rule:

- https://github.com/bazelbuild/bazel-skylib/pull/338
- https://github.com/bazelbuild/bazel-skylib/pull/339
- https://github.com/bazelbuild/bazel-skylib/pull/340
- https://github.com/bazelbuild/bazel-skylib/pull/341

These rules let you wrap a pre-built binary or script in a conventional binary
and test rule respectively. They fulfill the same goal as sh_binary and sh_test
do, but they run the wrapped binary directly, instead of through Bash, so they
don't depend on Bash and work with --shell_exectuable="".
"""

def _shared_impl(ctx):
    out = ctx.attr.out
    if not out:
        out = ctx.attr.name
    output = ctx.actions.declare_file(out)
    ctx.actions.symlink(
        target_file = ctx.executable.src,
        output = output,
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = ctx.files.data)

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = ([d[DefaultInfo].default_runfiles for d in ctx.attr.data] +
                [ctx.attr.src[DefaultInfo].default_runfiles])
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)
    return DefaultInfo(
        executable = output,
        files = depset([output]),
        runfiles = runfiles,
    )

def _native_binary_impl(ctx):
    default_info = _shared_impl(ctx)
    return [default_info]

def _native_test_impl(ctx):
    default_info = _shared_impl(ctx)
    return [default_info, testing.TestEnvironment(ctx.attr.env)]

# We have to manually set "env" on the test rule because the builtin one is only
# available in native rules. See
# https://docs.bazel.build/versions/main/be/common-definitions.html#test.env
# We don't have "env" on native_binary because there is no BinaryEnvironment
# mirroring TestEnvironment. See https://github.com/bazelbuild/bazel/issues/7364
_SHARED_ATTRS = {
    "src": attr.label(
        executable = True,
        allow_files = True,
        mandatory = True,
        cfg = "target",
    ),
    "data": attr.label_list(allow_files = True),
    # "out" is attr.string instead of attr.output, so that it is select()'able.
    "out": attr.string(),
}

native_binary = rule(
    implementation = _native_binary_impl,
    attrs = _SHARED_ATTRS,
    executable = True,
)

_TEST_ATTRS = {
    k: v
    for k, v in _SHARED_ATTRS.items() + [
        (
            "env",
            attr.string_dict(
                doc = "Mirrors the common env attribute that otherwise is" +
                      " only available on native rules. See" +
                      " https://docs.bazel.build/versions/main/be/common-definitions.html#test.env",
            ),
        ),
    ]
}

native_test = rule(
    implementation = _native_test_impl,
    attrs = _TEST_ATTRS,
    test = True,
)
