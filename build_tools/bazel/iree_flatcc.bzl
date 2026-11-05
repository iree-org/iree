# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates FlatBuffer source files with flatcc."""

load("@rules_cc//cc:defs.bzl", "cc_library")

def iree_flatbuffer_c_library(
        name,
        srcs,
        flatcc_args = ["--common", "--reader"],
        includes = [],
        testonly = False,
        **kwargs):
    flatcc = Label("@com_github_dvidelabs_flatcc//:flatcc")

    # "-I" flags are resolved by flatcc relative to working directory that
    # bazel uses to call it. So if iree is not the main module, but only a
    # dependency, the include dir needs to be adapted.
    repo_name = Label("//:MODULE.bazel").repo_name
    include_root = ("external/%s/runtime/src" % repo_name) if repo_name else "runtime/src"

    flags = [
        "-o$(RULEDIR)",
        "-I " + include_root,
    ] + flatcc_args

    out_stem = "%s" % (srcs[0].replace(".fbs", ""))

    outs = []
    for arg in flags:
        if arg == "--reader":
            outs.append("%s_reader.h" % (out_stem))
        if arg == "--builder":
            outs.append("%s_builder.h" % (out_stem))
        if arg == "--verifier":
            outs.append("%s_verifier.h" % (out_stem))
        if arg == "--json":
            outs.append("%s_json_parser.h" % (out_stem))
            outs.append("%s_json_printer.h" % (out_stem))

    native.genrule(
        name = name + "_gen",
        srcs = srcs + includes,
        outs = outs,
        tools = [flatcc],
        cmd = "$(location %s) %s %s" % (flatcc, " ".join(flags), " ".join(["$(location {})".format(src) for src in srcs])),
        testonly = testonly,
    )
    cc_library(
        name = name,
        hdrs = outs,
        testonly = testonly,
        **kwargs
    )
