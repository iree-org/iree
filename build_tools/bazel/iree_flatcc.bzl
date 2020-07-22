# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates flatbuffer source files with flatcc."""

def iree_flatbuffer_c_library(
        name,
        srcs,
        flatcc_args = ["--common", "--reader"],
        testonly = False,
        **kwargs):
    flatcc = "@com_github_dvidelabs_flatcc//:flatcc"
    flatcc_rt = "@com_github_dvidelabs_flatcc//:runtime"

    flags = [
        "-o$(RULEDIR)",
    ] + flatcc_args

    out_stem = "%s" % (srcs[0].replace(".fbs", ""))

    outs = []
    for arg in flags:
        if arg == "--reader":
            outs += ["%s_reader.h" % (out_stem)]
        if arg == "--builder":
            outs += ["%s_builder.h" % (out_stem)]
        if arg == "--verifier":
            outs += ["%s_verifier.h" % (out_stem)]

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
        deps = [
            flatcc_rt,
        ],
        testonly = testonly,
        **kwargs
    )
