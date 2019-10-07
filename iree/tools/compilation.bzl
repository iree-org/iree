"""Rules for compiling IREE executables, modules, and archives."""

load("//tools/build_defs/cc:cc_embed_data.bzl", "cc_embed_data")

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
                "--print-after-all",
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
            srcs = ["%s.emod" % (name)],
            outs = [
                # NOTE: we do not generate the .o as it is tricky on platforms
                # like wasm.
                "%s.cc" % (name),
                "%s.h" % (name),
            ],
            embedopts = ["--namespace=%s" % (cc_namespace)],
            visibility = visibility,
        )
