# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wasm JS companion library, entry point, and binary bundling rules.

iree_wasm_cc_library declares JS files that accompany a C library compiled for
wasm. When the C library declares wasm imports (via __attribute__((import_module))),
the corresponding JS implementations live in an iree_wasm_cc_library target.

iree_wasm_entry declares the JS entry point for a wasm binary. The entry point
orchestrates wasm instantiation (WASI setup, worker spawning, import merging).
Adding an iree_wasm_entry target to deps of a wasm binary or test causes it to
be discovered automatically — no explicit "main" parameter needed.

At binary link time, the collect_wasm_js aspect walks the transitive dependency
graph of a cc_binary, collecting all IreeWasmJsInfo and IreeWasmEntryInfo
providers. The bundler then:
  1. Parses the .wasm binary's import section to find required module names.
  2. Matches against collected modules to perform dead code elimination.
  3. Topologically sorts by dependency order (depset provides this naturally).
  4. Concatenates into a single .mjs file with the entry point.
"""

# --- Companion library (wasm import implementations) -----------------------

IreeWasmJsInfo = provider(
    doc = "JS companion files for a wasm C library.",
    fields = {
        "js_files": "depset of File objects containing JS companion sources.",
        "modules": "depset of structs (module: string, js_files: tuple of Files).",
    },
)

def _iree_wasm_cc_library_impl(ctx):
    js_files = ctx.files.srcs

    transitive_js = []
    transitive_modules = []
    for dep in ctx.attr.deps:
        transitive_js.append(dep[IreeWasmJsInfo].js_files)
        transitive_modules.append(dep[IreeWasmJsInfo].modules)

    module_info = struct(
        module = ctx.attr.module,
        js_files = tuple(js_files),
    )

    return [
        IreeWasmJsInfo(
            js_files = depset(js_files, transitive = transitive_js),
            modules = depset([module_info], transitive = transitive_modules),
        ),
        # Empty CcInfo so this target can appear in cc_library deps without
        # breaking the C compilation. The C compiler sees nothing from this
        # target; it exists purely to participate in the dependency graph.
        CcInfo(),
    ]

iree_wasm_cc_library = rule(
    implementation = _iree_wasm_cc_library_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".js", ".mjs"],
            doc = "JS source files implementing the wasm imports for this module.",
        ),
        "module": attr.string(
            mandatory = True,
            doc = "Wasm import module name (e.g., 'iree_syscall'). " +
                  "This matches the import_module attribute in the C headers.",
        ),
        "deps": attr.label_list(
            providers = [IreeWasmJsInfo],
            doc = "Other iree_wasm_cc_library targets this library depends on.",
        ),
    },
    provides = [IreeWasmJsInfo, CcInfo],
    doc = "Declares JS companion files for a C library compiled to wasm.",
)

# --- Entry point (JS orchestrator for wasm instantiation) ------------------

IreeWasmEntryInfo = provider(
    doc = "Entry point JS file for a wasm binary.",
    fields = {
        "main": "File object for the entry point .mjs file.",
        "srcs": "depset of File objects — local imports of the entry point. " +
                "The bundler resolves and inlines these at build time.",
    },
)

def _iree_wasm_entry_impl(ctx):
    return [
        IreeWasmEntryInfo(
            main = ctx.file.main,
            srcs = depset(ctx.files.srcs),
        ),
        # Empty CcInfo so this target can appear in cc_library deps without
        # breaking the C compilation, same as iree_wasm_cc_library.
        CcInfo(),
    ]

iree_wasm_entry = rule(
    implementation = _iree_wasm_entry_impl,
    attrs = {
        "main": attr.label(
            allow_single_file = [".js", ".mjs"],
            mandatory = True,
            doc = "The entry point JS file. This is the JS equivalent of " +
                  "main.cc — it orchestrates wasm instantiation using the " +
                  "generated createWasmImports() function.",
        ),
        "srcs": attr.label_list(
            allow_files = [".js", ".mjs"],
            doc = "Local imports of the entry point. The bundler resolves " +
                  "relative ESM imports at build time, so any files imported " +
                  "by the entry point via relative paths must be listed here.",
        ),
    },
    provides = [IreeWasmEntryInfo, CcInfo],
    doc = "Declares the JS entry point for a wasm binary. When this target " +
          "appears in deps of an iree_wasm_cc_binary, iree_wasm_cc_test, or " +
          "iree_cc_test, the entry point is discovered automatically.",
)

# --- Aspect for collecting JS info through cc_library deps -----------------

IreeWasmJsCollectionInfo = provider(
    doc = "Collected JS companion files from the transitive dependency graph.",
    fields = {
        "js_files": "depset of all transitive JS companion File objects.",
        "modules": "depset of all transitive module structs.",
    },
)

IreeWasmEntryCollectionInfo = provider(
    doc = "Collected entry points from the transitive dependency graph.",
    fields = {
        "entries": "depset of structs (main: File, srcs: tuple of Files).",
    },
)

def _collect_wasm_js_impl(target, ctx):
    transitive_js = []
    transitive_modules = []
    transitive_entries = []

    # If this target directly provides companion or entry info, include it.
    if IreeWasmJsInfo in target:
        transitive_js.append(target[IreeWasmJsInfo].js_files)
        transitive_modules.append(target[IreeWasmJsInfo].modules)
    if IreeWasmEntryInfo in target:
        entry = struct(
            main = target[IreeWasmEntryInfo].main,
            srcs = tuple(target[IreeWasmEntryInfo].srcs.to_list()),
        )
        transitive_entries.append(depset([entry]))

    # Collect from deps that have already been visited by this aspect.
    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if IreeWasmJsCollectionInfo in dep:
                transitive_js.append(dep[IreeWasmJsCollectionInfo].js_files)
                transitive_modules.append(dep[IreeWasmJsCollectionInfo].modules)
            if IreeWasmEntryCollectionInfo in dep:
                transitive_entries.append(dep[IreeWasmEntryCollectionInfo].entries)

    return [
        IreeWasmJsCollectionInfo(
            js_files = depset(transitive = transitive_js),
            modules = depset(transitive = transitive_modules),
        ),
        IreeWasmEntryCollectionInfo(
            entries = depset(transitive = transitive_entries),
        ),
    ]

collect_wasm_js = aspect(
    implementation = _collect_wasm_js_impl,
    attr_aspects = ["deps"],
    doc = "Walks the transitive dependency graph collecting IreeWasmJsInfo " +
          "and IreeWasmEntryInfo from targets reachable through cc_library deps.",
)

# --- Entry point discovery -------------------------------------------------

def _discover_entry(cc_deps):
    """Discovers a single iree_wasm_entry from the transitive deps.

    Args:
        cc_deps: List of targets with optional IreeWasmEntryCollectionInfo.

    Returns:
        A struct (main: File, srcs: tuple of Files), or None if no entry found.
        Fails if multiple entries are found.
    """
    entries = []
    for dep in cc_deps:
        if IreeWasmEntryCollectionInfo in dep:
            entries.extend(dep[IreeWasmEntryCollectionInfo].entries.to_list())
    if len(entries) > 1:
        fail("Multiple iree_wasm_entry targets found in deps; expected at most one")
    if entries:
        return entries[0]
    return None

# --- Wasm binary bundling ---------------------------------------------------

def collect_and_bundle(ctx, wasm_binary, main_js, cc_deps, bundler, main_srcs = []):
    """Collects JS companions from cc_deps and produces a bundled .mjs file.

    Walks cc_deps via the collect_wasm_js aspect, writes a modules manifest,
    and runs the bundler to produce a single .mjs file with the entry point.

    Args:
        ctx: Rule context (for actions and label).
        wasm_binary: File object for the .wasm binary.
        main_js: File object for the entry point JS file.
        cc_deps: List of targets with optional IreeWasmJsCollectionInfo.
        bundler: Executable for the wasm binary bundler.
        main_srcs: List of File objects for the entry point's local imports.
            The bundler resolves relative ESM imports at build time, so any
            files referenced by the entry point must be present in the sandbox.

    Returns:
        output_mjs: The bundled .mjs File.
    """
    transitive_modules = []
    for dep in cc_deps:
        if IreeWasmJsCollectionInfo in dep:
            transitive_modules.append(dep[IreeWasmJsCollectionInfo].modules)

    all_modules = depset(transitive = transitive_modules)
    all_js_files = depset(transitive = [
        depset(transitive = [
            dep[IreeWasmJsCollectionInfo].js_files
            for dep in cc_deps
            if IreeWasmJsCollectionInfo in dep
        ]),
    ])

    # Write the modules manifest as JSON for the bundler.
    modules_file = ctx.actions.declare_file(ctx.label.name + "_wasm_modules.json")
    module_entries = []
    for mod in all_modules.to_list():
        for js_file in mod.js_files:
            module_entries.append(json.encode({
                "module": mod.module,
                "path": js_file.path,
            }))
    ctx.actions.write(
        output = modules_file,
        content = "[" + ",".join(module_entries) + "]",
    )

    # Run the bundler.
    output_mjs = ctx.actions.declare_file(ctx.label.name + ".mjs")
    arguments = ctx.actions.args()
    arguments.add("--wasm", wasm_binary)
    arguments.add("--wasm-filename", wasm_binary.basename)
    arguments.add("--main", main_js)
    arguments.add("--modules", modules_file)
    arguments.add("--output", output_mjs)

    ctx.actions.run(
        executable = bundler,
        arguments = [arguments],
        inputs = depset(
            [wasm_binary, main_js, modules_file] + list(main_srcs),
            transitive = [all_js_files],
        ),
        outputs = [output_mjs],
        mnemonic = "IreeWasmBundle",
        progress_message = "Bundling JS companions for %s" % ctx.label,
    )

    return output_mjs

def _iree_wasm_bundle_impl(ctx):
    main_js = ctx.file.main
    main_srcs = []

    # If no explicit main, discover from the dependency graph.
    if main_js == None:
        entry = _discover_entry(ctx.attr.cc_deps)
        if entry == None:
            fail("No entry point: provide an explicit 'main' attribute or " +
                 "add an iree_wasm_entry target to deps")
        main_js = entry.main
        main_srcs = list(entry.srcs)

    output_mjs = collect_and_bundle(
        ctx = ctx,
        wasm_binary = ctx.file.binary,
        main_js = main_js,
        cc_deps = ctx.attr.cc_deps,
        bundler = ctx.executable._bundler,
        main_srcs = main_srcs,
    )
    return [DefaultInfo(
        files = depset([output_mjs]),
        runfiles = ctx.runfiles(files = [ctx.file.binary, output_mjs]),
    )]

_iree_wasm_bundle = rule(
    implementation = _iree_wasm_bundle_impl,
    attrs = {
        "binary": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The cc_binary target producing the .wasm file.",
        ),
        "main": attr.label(
            allow_single_file = [".js", ".mjs"],
            doc = "Entry point JS file. If omitted, discovered from cc_deps " +
                  "via an iree_wasm_entry target.",
        ),
        "cc_deps": attr.label_list(
            aspects = [collect_wasm_js],
            doc = "Same deps as the cc_binary, for aspect-based JS collection.",
        ),
        "_bundler": attr.label(
            default = "//build_tools/wasm:wasm_binary_bundler",
            executable = True,
            cfg = "exec",
        ),
    },
    doc = "Bundles JS companion files collected from the transitive dependency " +
          "graph of a wasm binary into a single .mjs file.",
)

# --- Public macros ----------------------------------------------------------

def iree_wasm_cc_binary(name, main = None, srcs = None, deps = None, **kwargs):
    """Creates a wasm binary with bundled JS companions.

    This macro creates three targets:
      - {name}_wasm: the raw cc_binary producing the .wasm file.
      - {name}_bundle: the bundle producing {name}_bundle.mjs alongside
          the .wasm. Use this target in filegroup/deploy rules to get
          clean outputs (just .mjs, no shell wrapper).
      - {name}: sh_binary wrapper. 'bazel run :{name}' executes via node.

    The bundler walks the transitive deps, collects all iree_wasm_cc_library
    targets, parses the .wasm import section for dead code elimination, and
    concatenates the JS companions in dependency order with the entry point.

    The entry point is either specified explicitly via 'main' or discovered
    from an iree_wasm_entry target in deps.

    Args:
        name: Target name. 'bazel run :{name}' runs via node.
        main: Entry point JS file (label or select()). If omitted,
            an iree_wasm_entry target must be present in deps.
        srcs: C/C++ source files for the wasm binary.
        deps: Dependencies (both C libraries and iree_wasm_cc_library targets).
        **kwargs: Additional arguments passed to cc_binary (e.g.,
            target_compatible_with, copts, linkopts).
    """
    native.cc_binary(
        name = name + "_wasm",
        srcs = srcs or [],
        deps = deps or [],
        **kwargs
    )

    bundle_kwargs = {
        "binary": ":" + name + "_wasm",
        "cc_deps": deps or [],
    }
    if main:
        bundle_kwargs["main"] = main

    _iree_wasm_bundle(
        name = name + "_bundle",
        **bundle_kwargs
    )

    native.sh_binary(
        name = name,
        srcs = ["//build_tools/wasm:wasm32_node_test_runner.sh"],
        data = [":" + name + "_bundle"],
        args = ["$(rootpath :" + name + "_bundle)"],
    )

def iree_wasm_cc_test(name, main = None, srcs = None, deps = None, **kwargs):
    """Creates a wasm test binary with bundled JS companions.

    Same internal structure as iree_wasm_cc_binary, but the user-facing
    target is an sh_test instead of sh_binary. The test passes if the
    .mjs entry point exits with code 0.

    This macro creates three targets:
      - {name}_wasm: the raw cc_binary producing the .wasm file.
      - {name}_bundle: the bundle producing {name}_bundle.mjs.
      - {name}: sh_test that runs the bundle via node.

    Args:
        name: Target name. 'bazel test :{name}' runs the test.
        main: Entry point JS file. If omitted, an iree_wasm_entry target
            must be present in deps.
        srcs: C/C++ source files for the wasm binary.
        deps: Dependencies (both C libraries and iree_wasm_cc_library targets).
        **kwargs: Additional arguments passed to cc_binary.
    """

    # Test binaries must be testonly to depend on testonly libraries.
    kwargs.pop("testonly", None)

    native.cc_binary(
        name = name + "_wasm",
        srcs = srcs or [],
        deps = deps or [],
        testonly = True,
        **kwargs
    )

    bundle_kwargs = {
        "testonly": True,
        "binary": ":" + name + "_wasm",
        "cc_deps": deps or [],
    }
    if main:
        bundle_kwargs["main"] = main

    _iree_wasm_bundle(
        name = name + "_bundle",
        **bundle_kwargs
    )

    native.sh_test(
        name = name,
        srcs = ["//build_tools/wasm:wasm32_node_test_runner.sh"],
        data = [":" + name + "_bundle"],
        args = ["$(rootpath :" + name + "_bundle)"],
    )
