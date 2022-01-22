# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for running lit tests with the upstream lit binary."""

# This exists as a separate file from iree_lit_test.bzl because we anticipate
# upstreaming it soon.

load("@bazel_skylib//lib:paths.bzl", "paths")
load(":native_binary.bzl", "native_test")

def _tools_on_path_impl(ctx):
    runfiles = ctx.runfiles()

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = [d[DefaultInfo].default_runfiles for d in ctx.attr.srcs]
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)

    runfiles_symlinks = {}

    for src in ctx.attr.srcs:
        exe = src[DefaultInfo].files_to_run.executable
        if not exe:
            fail("All targets used as tools by lit tests must have exactly one" +
                 " executable, but {} has none".format(src))
        bin_path = paths.join(ctx.attr.bin_dir, exe.basename)
        if bin_path in runfiles_symlinks:
            fail("All tools used by lit tests must have unique basenames, as" +
                 " they are added to the path." +
                 " {} and {} conflict".format(runfiles_symlinks[bin_path], exe))
        runfiles_symlinks[bin_path] = exe

    return [
        DefaultInfo(runfiles = ctx.runfiles(
            symlinks = runfiles_symlinks,
        ).merge(runfiles)),
    ]

_tools_on_path = rule(
    _tools_on_path_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "bin_dir": attr.string(mandatory = True),
    },
    doc = "Symlinks srcs into a single lit_bin directory. All basenames must be unique.",
)

def lit_test(
        name,
        test_file,
        cfg,
        tools = None,
        args = None,
        data = None,
        visibility = None,
        env = None,
        **kwargs):
    """Runs a single test file with LLVM's lit tool.

    Args:
      name: string. the name of the generated test target.
      test_file: label. The file on which to run lit.
      cfg: label. The lit config file. It must list the file extension of
        `test_file` in config.suffixes and must be in a parent directory of
        `test_file`.
      tools: label list. Tools invoked in the lit RUN lines. These binaries will
        be symlinked into a directory which is on the path. They must therefore
        have unique basenames.
      args: string list. Additional arguments to pass to lit. Note that the test
        file, `-v`, and a `--path` argument for the directory to which `tools`
        are symlinked are added automatically.
      data: label list. Additional data dependencies of the test. Note that
        targets in `cfg` and `tools`, as well as their data dependencies, are
        added automatically.
      visibility: visibility of the generated test target.
      env: string_dict. Environment variables available during test execution.
        See the common Bazel test attribute.
      **kwargs: additional keyword arguments to pass to all generated rules.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit
    """
    args = args or []
    data = data or []
    tools = tools or []

    tools_on_path_target_name = "_{}_tools_on_path".format(name)

    bin_dir = paths.join(
        native.package_name(),
        tools_on_path_target_name,
        "lit_bin",
    )

    _tools_on_path(
        name = tools_on_path_target_name,
        testonly = True,
        srcs = tools,
        bin_dir = bin_dir,
        visibility = ["//visibility:private"],
        **kwargs
    )

    native_test(
        name = name,
        src = "@llvm-project//llvm:lit",
        # out = name,
        args = [
            "-v",
            "--path",
            bin_dir,
            "$(location {})".format(test_file),
        ] + args,
        data = [test_file, cfg, tools_on_path_target_name] + data,
        visibility = visibility,
        env = env,
        **kwargs
    )

def lit_test_suite(
        name,
        srcs,
        cfg,
        tools = None,
        args = None,
        data = None,
        visibility = None,
        size = "small",
        env = None,
        **kwargs):
    """Creates one lit test per source file and a test suite that bundles them.

    Args:
      name: string. the name of the generated test suite.
      srcs: label_list. The files which contain the lit tests.
      cfg: label. The lit config file. It must list the file extension of
        the files in `srcs` in config.suffixes and must be in a parent directory
        of `srcs`.
      tools: label list. Tools invoked in the lit RUN lines. These binaries will
        be symlinked into a directory which is on the path. They must therefore
        have unique basenames.
      args: string list. Additional arguments to pass to lit. Note that the test
        file, `-v`, and a `--path` argument for the directory to which `tools`
        are symlinked are added automatically.
      data: label list. Additional data dependencies of the test. Note that
        targets in `cfg` and `tools`, as well as their data dependencies, are
        added automatically.
      visibility: visibility of the generated test targets and test suite.
      size: string. size of the generated tests.
      env: string_dict. Environment variables available during test execution.
        See the common Bazel test attribute.
      **kwargs: additional keyword arguments to pass to all generated rules.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit
    """
    # If there are kwargs that need to be passed to only some of the generated
    # rules, they should be extracted into separate named arguments.

    args = args or []
    data = data or []
    tools = tools or []

    tests = []
    for test_file in srcs:
        # It's generally good practice to prefix any generated names with the
        # macro name, but it's also nice to have the test name just match the
        # file name.
        test_name = "%s.test" % (test_file)
        tests.append(test_name)
        lit_test(
            name = test_name,
            test_file = test_file,
            cfg = cfg,
            tools = tools,
            args = args,
            data = data,
            visibility = visibility,
            env = env,
            **kwargs
        )

    native.test_suite(
        name = name,
        tests = tests,
        **kwargs
    )
