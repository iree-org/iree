# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ provider transforms for the renamed IREE compiler ABI.

The compiler library renames every llvm/mlir C++ symbol before link (see
gen_rename_map.py), so anything linking against it must rename to match,
undefined references included.

Rewritten archives are declared as new provider values rather than mutating the
inputs, keeping Bazel's C++ provider graph intact.

External plugins want iree_compiler_register_dynamic_plugin. The ABI target and
CcInfo transform stay separate so each can be tested alone.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

RenamedCompilerAbiInfo = provider(fields = {
    "symbol_prefix": "Logical ABI prefix inserted as an Itanium component.",
})

def _bzlmod_repo_name_matches(workspace_name, repo_name):
    return (
        workspace_name.endswith("+" + repo_name) or
        workspace_name.endswith("~" + repo_name)
    )

def _provided_by_compiler(owner, compiler_workspace_name):
    workspace_name = owner.workspace_name
    owner_text = str(owner)

    if "llvm-project" in workspace_name or "llvm-project" in owner_text:
        return True

    # libbacktrace is already in the dylib and exports plain C, so let consumers
    # resolve it there rather than carry every platform backend themselves.
    if "libbacktrace" in workspace_name or "libbacktrace" in owner_text:
        return True

    # Whatever repository the compiler target came from is inside the dylib
    # already. In-tree that is the main workspace; from a plugin repository it
    # is @iree_core, and there the plugin's own main-workspace archives still
    # need renaming. The caller keeps its direct deps regardless, so an in-tree
    # plugin sharing the main workspace with the compiler is not mistaken for
    # part of it.
    if compiler_workspace_name != None and workspace_name == compiler_workspace_name:
        return True

    # Fallbacks for a consumer that names no compiler target.
    if workspace_name == "iree_core":
        return True
    return _bzlmod_repo_name_matches(workspace_name, "iree_core") or \
           _bzlmod_repo_name_matches(workspace_name, "iree")

def _artifact_suffix(file, suffixes, description):
    path = file.path
    for suffix in suffixes:
        if path.endswith(suffix):
            return suffix
    fail("Expected {}, got {}".format(description, file.path))

def _sanitize_path_fragment(value):
    """Returns a stable file-name fragment.

    Example: `@llvm-project//mlir:IR.pic.a` becomes
    `_llvm_project__mlir_IR_pic_a`, which is safe to use in declared outputs.
    """
    result = ""
    for i in range(len(value)):
        c = value[i]
        if c.isalnum() or c == "_":
            result += c
        else:
            result += "_"
    return result

def _rename_file(ctx, input_file, output_base, symbol_prefix, defined_only, suffixes, description):
    suffix = _artifact_suffix(input_file, suffixes, description)
    output_base = _sanitize_path_fragment(output_base)
    out = ctx.actions.declare_file(
        "{}/{}{}".format(ctx.label.name + "_renamed", output_base, suffix),
    )
    map_file = ctx.actions.declare_file(
        "{}/{}.rename_map".format(ctx.label.name + "_renamed", output_base),
    )

    map_args = ctx.actions.args()
    map_args.add("--nm", ctx.executable._nm)
    map_args.add("--cxxfilt", ctx.executable._cxxfilt)
    map_args.add("--input", input_file)
    map_args.add("--out", map_file)
    map_args.add("--symbol-prefix", symbol_prefix)
    if defined_only:
        map_args.add("--defined-only")
    ctx.actions.run(
        executable = ctx.executable._gen_map,
        arguments = [map_args],
        inputs = [input_file],
        tools = [ctx.executable._nm, ctx.executable._cxxfilt],
        outputs = [map_file],
        mnemonic = "RenameMap",
        progress_message = "Computing llvm/mlir rename map for %s" % input_file.short_path,
    )

    objcopy_args = ctx.actions.args()
    objcopy_args.add("--redefine-syms=" + map_file.path)
    objcopy_args.add(input_file)
    objcopy_args.add(out)
    ctx.actions.run(
        executable = ctx.executable._objcopy,
        arguments = [objcopy_args],
        inputs = [input_file, map_file],
        outputs = [out],
        mnemonic = "RenameSymbols",
        progress_message = "Renaming llvm/mlir symbols in %s" % input_file.short_path,
    )

    return out

def _rename_archive(ctx, archive, output_base, symbol_prefix, defined_only):
    return _rename_file(
        ctx,
        archive,
        output_base,
        symbol_prefix,
        defined_only,
        [".pic.a", ".lo", ".a", ".lib"],
        "a static archive-like file",
    )

def _archive_objects(ctx, objects, output_base):
    if len(objects) == 0:
        return None
    output_base = _sanitize_path_fragment(output_base)
    out = ctx.actions.declare_file(
        "{}/{}.a".format(ctx.label.name + "_renamed", output_base),
    )
    args = ctx.actions.args()
    args.add("rcs")
    args.add(out)
    args.add_all(objects)
    ctx.actions.run(
        executable = ctx.executable._llvm_ar,
        arguments = [args],
        inputs = objects,
        outputs = [out],
        mnemonic = "ArchiveObjects",
        progress_message = "Archiving object-file linker input for %s" % ctx.label.name,
    )
    return out

def _has_objects(value):
    return value != None and len(value) != 0

def _library_identity(library):
    """Returns a de-duplication key, or None for flag/input-only libraries."""
    parts = []
    for artifact in [
        library.static_library,
        library.pic_static_library,
        library.dynamic_library,
        library.interface_library,
    ]:
        if artifact != None:
            parts.append(artifact.path)
    if _has_objects(library.objects):
        for artifact in library.objects:
            parts.append(artifact.path)
    if _has_objects(library.pic_objects):
        for artifact in library.pic_objects:
            parts.append(artifact.path)
    if len(parts) == 0:
        return None
    return "|".join(parts)

def _rename_library_to_link(
        ctx,
        cc_toolchain,
        feature_configuration,
        library,
        output_base,
        symbol_prefix,
        defined_only,
        force_alwayslink):
    static_library = None
    pic_static_library = None
    object_static_library = None
    object_pic_static_library = None

    # llvm-objcopy takes one file, so bare object lists are archived first.
    if library.static_library:
        static_library = _rename_archive(
            ctx,
            library.static_library,
            output_base + "_static",
            symbol_prefix,
            defined_only,
        )
    if library.pic_static_library:
        pic_static_library = _rename_archive(
            ctx,
            library.pic_static_library,
            output_base + "_pic_static",
            symbol_prefix,
            defined_only,
        )
    if _has_objects(library.objects) and library.static_library == None:
        object_archive = _archive_objects(ctx, library.objects, output_base + "_objects")
        object_static_library = _rename_archive(
            ctx,
            object_archive,
            output_base + "_objects_renamed",
            symbol_prefix,
            defined_only,
        )
    if _has_objects(library.pic_objects) and library.pic_static_library == None:
        object_pic_archive = _archive_objects(ctx, library.pic_objects, output_base + "_pic_objects")
        object_pic_static_library = _rename_archive(
            ctx,
            object_pic_archive,
            output_base + "_pic_objects_renamed",
            symbol_prefix,
            defined_only,
        )

    # Nothing to rewrite.
    renamed_files = [
        f
        for f in [
            static_library,
            pic_static_library,
            object_static_library,
            object_pic_static_library,
        ]
        if f != None
    ]
    if not renamed_files:
        return [library], []

    alwayslink = force_alwayslink or library.alwayslink

    # create_library_to_link mangles its input, so it rejects the _solib_ symlink
    # Bazel already mangled. Give it the original.
    dynamic_library = library.resolved_symlink_dynamic_library or library.dynamic_library
    interface_library = library.resolved_symlink_interface_library or library.interface_library

    libraries = []

    # Recreate the same static/dynamic shape the input had.
    if (
        static_library != None or
        pic_static_library != None or
        dynamic_library != None or
        interface_library != None
    ):
        libraries.append(cc_common.create_library_to_link(
            actions = ctx.actions,
            static_library = static_library,
            pic_static_library = pic_static_library,
            dynamic_library = dynamic_library,
            interface_library = interface_library,
            cc_toolchain = cc_toolchain if dynamic_library != None or interface_library != None else None,
            feature_configuration = feature_configuration if dynamic_library != None or interface_library != None else None,
            alwayslink = alwayslink,
        ))
    if object_static_library != None or object_pic_static_library != None:
        libraries.append(cc_common.create_library_to_link(
            actions = ctx.actions,
            static_library = object_static_library,
            pic_static_library = object_pic_static_library,
            alwayslink = alwayslink,
        ))
    return libraries, renamed_files

def _renamed_cc_info_impl(ctx):
    # Normally taken from the ABI target so plugins and the compiler agree. The
    # attr keeps this usable alone, in tests or standalone wiring.
    symbol_prefix = ctx.attr.symbol_prefix
    compiler_workspace_name = None
    if ctx.attr.compiler:
        symbol_prefix = ctx.attr.compiler[RenamedCompilerAbiInfo].symbol_prefix
        compiler_workspace_name = ctx.attr.compiler.label.workspace_name
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    direct_owners = {str(dep.label): True for dep in ctx.attr.deps}
    excluded_owners = {str(dep.label): True for dep in ctx.attr.exclude}
    linker_inputs = []
    renamed_files = []
    seen_libraries = {}
    library_ordinal = 0

    for dep in ctx.attr.deps:
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            owner = linker_input.owner
            owner_label = str(owner)
            if owner_label in excluded_owners:
                continue

            # Leaf deps only; their MLIR closure still comes from the dylib.
            if ctx.attr.direct_only and owner_label not in direct_owners:
                continue

            # Direct deps are the plugin's own code, renamed even when they
            # share a repository with the compiler.
            if (not ctx.attr.include_provided and
                owner_label not in direct_owners and
                _provided_by_compiler(owner, compiler_workspace_name)):
                continue

            libraries = []
            for library in linker_input.libraries:
                library_id = _library_identity(library)

                # Flags-only entry. Keep going so those fields survive.
                if library_id != None:
                    if library_id in seen_libraries:
                        continue
                    seen_libraries[library_id] = True

                renamed_libraries, files = _rename_library_to_link(
                    ctx,
                    cc_toolchain,
                    feature_configuration,
                    library,
                    "lib_{}".format(library_ordinal),
                    symbol_prefix,
                    ctx.attr.defined_only,
                    ctx.attr.force_alwayslink,
                )
                library_ordinal += 1
                libraries.extend(renamed_libraries)
                renamed_files.extend(files)

            if libraries or linker_input.user_link_flags or linker_input.additional_inputs:
                linker_inputs.append(cc_common.create_linker_input(
                    owner = owner,
                    libraries = depset(direct = libraries),
                    user_link_flags = depset(direct = linker_input.user_link_flags),
                    additional_inputs = depset(direct = linker_input.additional_inputs),
                ))

    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(direct = linker_inputs, order = "topological"),
    )
    return [
        DefaultInfo(files = depset(renamed_files)),
        CcInfo(linking_context = linking_context),
    ]

renamed_cc_info = rule(
    implementation = _renamed_cc_info_impl,
    doc = (
        "Low-level CcInfo transform that rewrites selected static archives to " +
        "the IREE compiler ABI. Prefer iree_compiler_register_dynamic_plugin for plugin BUILD files."
    ),
    attrs = {
        "deps": attr.label_list(providers = [CcInfo], default = []),
        "compiler": attr.label(
            providers = [[CcInfo, RenamedCompilerAbiInfo]],
            default = None,
            doc = "Optional compiler ABI target that supplies the symbol prefix.",
        ),
        "exclude": attr.label_list(
            providers = [CcInfo],
            default = [],
            doc = "Linker-input owners to omit entirely.",
        ),
        "include_provided": attr.bool(
            default = False,
            doc = "Keep archives from repos already provided by libIREECompiler.",
        ),
        "direct_only": attr.bool(
            default = False,
            doc = "Keep only linker inputs whose owner is one of deps' direct labels.",
        ),
        "defined_only": attr.bool(
            default = False,
            doc = "Rename only defined symbols in each archive.",
        ),
        "force_alwayslink": attr.bool(
            default = False,
            doc = "Force replacement static archives into whole-archive linking.",
        ),
        "symbol_prefix": attr.string(default = "IREE18"),
        "_nm": attr.label(default = "@llvm-project//llvm:llvm-nm", executable = True, cfg = "exec"),
        "_cxxfilt": attr.label(default = "@llvm-project//llvm:llvm-cxxfilt", executable = True, cfg = "exec"),
        "_objcopy": attr.label(default = "@llvm-project//llvm:llvm-objcopy", executable = True, cfg = "exec"),
        "_llvm_ar": attr.label(default = "@llvm-project//llvm:llvm-ar", executable = True, cfg = "exec"),
        "_gen_map": attr.label(default = "//build_tools/bazel:gen_rename_map", executable = True, cfg = "exec"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)

def _renamed_compiler_abi_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    library_to_link = cc_common.create_library_to_link(
        actions = ctx.actions,
        dynamic_library = ctx.file.shared_library,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
    )
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = [library_to_link]),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(direct = [linker_input]),
    )
    return [
        DefaultInfo(files = depset([ctx.file.shared_library])),
        RenamedCompilerAbiInfo(symbol_prefix = ctx.attr.symbol_prefix),
        CcInfo(linking_context = linking_context),
    ]

iree_renamed_compiler_abi = rule(
    implementation = _renamed_compiler_abi_impl,
    doc = "Wraps the renamed libIREECompiler shared library and its ABI prefix.",
    attrs = {
        "shared_library": attr.label(allow_single_file = True, mandatory = True),
        "symbol_prefix": attr.string(default = "IREE18"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
)

def iree_compiler_register_dynamic_plugin(plugin_id, target, compiler, extra_deps = [], linkopts = [], **kwargs):
    """Builds a compiler plugin, by id, against the renamed compiler ABI.

    The dynamic counterpart of iree_compiler_register_plugin. Spelled the same
    in both build systems so bazel_to_cmake can carry a declaration across.

    target is whole-archive linked: entry points are found by dlsym, not by
    references from the link.

    extra_deps are plugin-local MLIR archives libIREECompiler lacks. CMake
    cannot express them, so such a plugin will not convert.
    """

    # Same module name CMake emits.
    name = "iree_compiler_plugin_" + plugin_id
    renamed_cc_info(
        name = name + "_renamed_deps",
        deps = [target],
        compiler = compiler,
        force_alwayslink = True,
    )

    # The compiler is not linked in, only renamed against: its symbols stay
    # undefined and resolve from whichever host dlopens the plugin. Linking it
    # would give the plugin a second copy of the compiler, and with it a second
    # llvm::cl registry the host never parses.
    plugin_deps = [":" + name + "_renamed_deps"]
    if extra_deps:
        renamed_cc_info(
            name = name + "_renamed_extra_deps",
            deps = extra_deps,
            compiler = compiler,
            direct_only = True,
            include_provided = True,
        )
        plugin_deps.append(":" + name + "_renamed_extra_deps")

    native.cc_binary(
        name = name,
        srcs = [],
        linkshared = True,
        linkopts = linkopts + select({
            # ELF leaves undefined symbols alone; ld64 has to be told.
            "@platforms//os:macos": ["-Wl,-undefined,dynamic_lookup"],
            "//conditions:default": [],
        }),
        deps = plugin_deps,
        **kwargs
    )
