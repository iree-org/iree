# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates FlatBuffer source files with flatcc."""

def iree_flatbuffer_c_library(
        name,
        srcs,
        flatcc_args = ["--common", "--reader"],
        testonly = False,
        **kwargs):
    flatcc = "@com_github_dvidelabs_flatcc//:flatcc"

    flags = [
        "-o$(RULEDIR)",
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
        srcs = srcs,
        outs = outs,
        tools = [flatcc],
        cmd = "$(location %s) %s $(SRCS)" % (flatcc, " ".join(flags)),
        testonly = testonly,
    )
    native.cc_library(
        name = name,
        hdrs = outs,
        testonly = testonly,
        **kwargs
    )
