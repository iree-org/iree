"""Rules for compiling IREE executables, modules, and archives."""

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

# TODO(benvanik): port to a full starlark rule, document, etc.
def iree_module(
        name,
        srcs,
        cc_namespace = None,
        visibility = None):
    native.genrule(
        name = name,
        srcs = srcs,
        outs = [
            "%s.emod" % (name),
        ],
        cmd = " && ".join([
            " ".join([
                "$(location //iree/tools:iree-translate)",
                "-mlir-to-iree-module",
                "-o $(location %s.emod)" % (name),
            ] + ["$(locations %s)" % (src) for src in srcs]),
        ]),
        tools = [
            "//iree/tools:iree-translate",
        ],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
    )

    # Embed the module for use in C++. This avoids the need for file IO in
    # tests and samples that would otherwise complicate execution/porting.
    if cc_namespace:
        cc_embed_data(
            name = "%s_cc" % (name),
            identifier = name,
            srcs = ["%s.emod" % (name)],
            cc_file_output = "%s.cc" % (name),
            h_file_output = "%s.h" % (name),
            cpp_namespace = cc_namespace,
            visibility = visibility,
            flatten = True,
        )
