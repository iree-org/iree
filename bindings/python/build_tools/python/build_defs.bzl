# Copyright 2019 Google LLC
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

"""Macros for building native python extensions."""

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library")

def py_extension(
        name,
        srcs = [],
        data = [],
        copts = [],
        linkopts = [],
        linkstatic = 0,
        deps = [],
        features = [],
        win_def_file = None,
        **kwargs):
    """Builds a platform specific native python extension shared library.

    Note that you typically need to add some dependency on the python headers,
    which can typically be found in this repository as :python_headers.

    Args:
        name: Name of the final filegroup containing the shared library.
        srcs: cc_library srcs files to compile.
        data: cc_library data.
        copts: cc_library copts.
        linkopts: cc_library linkopts.
        linkstatic: cc_libarary linkstatic.
        deps: cc_library deps.
        features: cc_library features.
        win_def_file: The Windows DEF file to be passed to the linker.
        **kwargs: Any additional arguments to pass into all generated rules.
    """
    dll_file = name + ".dll"
    pyd_file = name + ".pyd"
    so_file = name + ".so"
    for platform_so_name in [dll_file, so_file]:
        actual_def_file = win_def_file if platform_so_name == dll_file else None
        cc_binary(
            name = platform_so_name,
            srcs = srcs,
            data = data,
            copts = copts,
            linkopts = linkopts,
            deps = deps,
            linkshared = True,
            linkstatic = linkstatic,
            features = features,
            win_def_file = actual_def_file,
            **kwargs
        )

    # TODO(laurenzo): Bug the bazel team about letting a cc_binary output
    # shared binaries with arbitrary extensions.
    native.genrule(
        name = pyd_file + "__pyd_copy",
        srcs = [":" + dll_file],
        outs = [":" + pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        **kwargs
    )

    # Making the rule output be a filegroup lets us give the shared
    # library a platform specific name (and maintain platform specific
    # rules if needed).
    filegroup_name = name + "__shared_objects"
    native.filegroup(
        name = filegroup_name,
        data = select({
            "@iree_native_python//:config_windows_any": [
                ":%s" % pyd_file,
            ],
            "//conditions:default": [
                ":%s" % so_file,
            ],
        }),
        **kwargs
    )
    py_library(
        name = name,
        data = [filegroup_name],
        **kwargs
    )
