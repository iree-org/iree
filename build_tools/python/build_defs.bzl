"""Macros for building native python extensions."""

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library")

def py_extension(
        name,
        srcs = [],
        data = [],
        copts = [],
        linkopts = [],
        deps = [],
        features = [],
        visibility = []):
    """Builds a platform specific native python extension shared library.

    Note that you typically need to add some dependency on the python headers,
    which can typically be found in this repository as :python_headers.

    Args:
        name: Name of the final filegroup containing the shared library.
        srcs: cc_library srcs files to compile.
        data: cc_library data.
        copts: cc_library copts.
        linkopts: cc_library linkopts.
        deps: cc_library deps.
        features: cc_library features.
        visibility: visibility for all artifacts.
    """
    dll_file = name + ".dll"
    pyd_file = name + ".pyd"
    so_file = name + ".so"
    for platform_so_name in [dll_file, so_file]:
        cc_binary(
            name = platform_so_name,
            srcs = srcs,
            data = data,
            copts = copts,
            linkopts = linkopts,
            deps = deps,
            linkshared = True,
            features = features,
            visibility = visibility,
        )

    # TODO(laurenzo): Bug the bazel team about letting a cc_binary output
    # shared binaries with arbitrary extensions.
    native.genrule(
        name = pyd_file + "__pyd_copy",
        srcs = [":" + dll_file],
        outs = [":" + pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
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
        visibility = visibility,
    )
    py_library(
        name = name,
        data = [filegroup_name],
    )
