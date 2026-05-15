# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Creates a clang-based linker driver for wasm32.

Bazel's cc_toolchain generates linker flags using GCC driver conventions
(-Wl,X prefix, -shared for shared libraries). Using clang as the linker
driver handles these conventions natively — no flag translation needed.

Clang discovers the wasm linker by searching -B directories for a binary
named "wasm-ld". This rule bundles a wasm-ld symlink (pointing at lld)
and creates a script that invokes clang with -B pointing at that directory.

The `extra_flags` attribute provides target-specific flags that are baked into
the wrapper script (e.g., --target=wasm32-wasi vs --target=wasm32-unknown-unknown).
Flags that reference execroot-relative paths (--sysroot, -resource-dir)
should go in cc_args instead, since the wrapper script's working directory
is NOT the execroot.

The `suffix_flags` attribute provides flags appended AFTER all Bazel-generated
arguments ("$@"). This controls link order for system libraries: Bazel places
toolchain cc_args flags before user objects/archives, but system libraries
(-lc++, -lwasi-emulated-signal) must come after user archives for the linker's
single-pass archive extraction to resolve forward references (e.g., libc's
__main_void creating a weak ref to __main_argc_argv, which gtest_main defines).

The `filter_flags` attribute lists flags to strip from Bazel's params files
before invoking clang. Bazel's built-in features inject host-oriented flags
(e.g., -pthread, -pie) that can be harmful for wasm targets — -pthread
enables --shared-memory which requires atomics-compiled sysroot libraries.
"""

def _wasm_clang_linker_impl(ctx):
    clang = ctx.executable.clang
    lld = ctx.executable.lld

    # Both clang and wasm-ld must be co-located so the wrapper script can
    # invoke clang and clang can find wasm-ld, using a single -B path.
    tool_dir = ctx.attr.name + "_dir"

    clang_link = ctx.actions.declare_file(tool_dir + "/clang")
    ctx.actions.symlink(
        output = clang_link,
        target_file = clang,
        is_executable = True,
    )

    wasm_ld = ctx.actions.declare_file(tool_dir + "/wasm-ld")
    ctx.actions.symlink(
        output = wasm_ld,
        target_file = lld,
        is_executable = True,
    )

    extra_args = " ".join(ctx.attr.extra_flags)
    suffix_args = " ".join(ctx.attr.suffix_flags)

    # Generate the filter block if filter_flags is set. For each @params_file
    # argument, strip matching lines before clang processes them.
    if ctx.attr.filter_flags:
        # Build a sed expression that deletes exact-match lines for each flag.
        # e.g., -e '/^-pthread$/d' -e '/^-pie$/d'
        sed_expressions = " ".join([
            "-e '/^{}$/d'".format(flag)
            for flag in ctx.attr.filter_flags
        ])
        filter_block = """\
# Strip flags that Bazel's built-in features inject but are harmful for
# wasm targets. Operates on @params files since Bazel passes link flags
# that way rather than as individual arguments.
for arg in "$@"; do
  case "$arg" in
    @*) sed {sed_expressions} "${{arg#@}}" > "${{arg#@}}.tmp" && mv "${{arg#@}}.tmp" "${{arg#@}}" ;;
  esac
done
""".format(sed_expressions = sed_expressions)
    else:
        filter_block = ""

    # The wrapper invokes clang as a WebAssembly linker driver.
    # -B tells clang where to find wasm-ld (searched before PATH).
    # All Bazel-generated flags (-Wl,X, -shared, @paramfile) pass through
    # unchanged — clang handles GCC driver conventions natively.
    #
    # -Wno-unused-command-line-argument suppresses warnings from
    # host-oriented flags that Bazel's built-in features inject
    # (e.g. -pie from force_pic_flags) that don't apply to wasm.
    wrapper = ctx.actions.declare_file(ctx.attr.name)
    ctx.actions.write(
        output = wrapper,
        is_executable = True,
        content = """\
#!/bin/sh
DIR="$(dirname "$0")/{tool_dir}"
{filter_block}exec "$DIR/clang" {extra_args} -fuse-ld=lld \
  -Wno-unused-command-line-argument -B "$DIR" "$@" {suffix_args}
""".format(
            tool_dir = tool_dir,
            extra_args = extra_args,
            filter_block = filter_block,
            suffix_args = suffix_args,
        ),
    )

    return [DefaultInfo(
        executable = wrapper,
        runfiles = ctx.runfiles(files = [clang_link, wasm_ld, clang, lld]),
    )]

wasm_clang_linker = rule(
    implementation = _wasm_clang_linker_impl,
    attrs = {
        "clang": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
            doc = "The clang binary to use as the linker driver.",
        ),
        "lld": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
            doc = "The lld binary, symlinked as wasm-ld for clang to discover.",
        ),
        "extra_flags": attr.string_list(
            default = [],
            doc = "Extra flags baked into the wrapper (target triple, -nostdlib, etc.).",
        ),
        "suffix_flags": attr.string_list(
            default = [],
            doc = "Flags appended after all Bazel-generated arguments (system libraries).",
        ),
        "filter_flags": attr.string_list(
            default = [],
            doc = "Flags to strip from Bazel params files before invoking clang.",
        ),
    },
    executable = True,
    doc = "Creates a clang-based linker driver that handles GCC driver conventions natively for wasm32.",
)
