# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Creates a test from the binary output of another rule.

The rule instantiation can pass additional arguments to the binary and provide
it with additional data files (as well as the standard bazel test classification
attributes). This allows compiling the binary once and not recompiling or
relinking it for each test rule. It also avoids a wrapper shell script, which
adds unnecessary shell dependencies and confuses some tooling about the type of
the binary.

Example usage:

run_binary_test(
    name = "my_test",
    args = ["--module_file=$(location :data_file)"],
    data = [":data_file"],
    test_binary = ":some_cc_binary",
)
"""

def _run_binary_test_impl(ctx):
    ctx.actions.symlink(
        target_file = ctx.executable.test_binary,
        output = ctx.outputs.executable,
        is_executable = True,
    )

    data_runfiles = ctx.runfiles(files = ctx.files.data)

    binary_runfiles = ctx.attr.test_binary[DefaultInfo].default_runfiles

    return [DefaultInfo(
        executable = ctx.outputs.executable,
        runfiles = data_runfiles.merge(binary_runfiles),
    )]

run_binary_test = rule(
    _run_binary_test_impl,
    attrs = {
        "test_binary": attr.label(
            mandatory = True,
            executable = True,
            cfg = "target",
        ),
        "data": attr.label_list(allow_empty = True, allow_files = True),
    },
    test = True,
)
