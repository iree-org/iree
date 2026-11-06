# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C/C++ toolchain configuration for Android aarch64 using the Android NDK."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)
load(
    "@iree_android_ndk//:defs.bzl",
    "ANDROID_API_LEVEL",
    "ANDROID_NDK_AVAILABLE",
    "ANDROID_NDK_ERROR",
    "ANDROID_NDK_LIB_CLANG_DIR",
    "ANDROID_NDK_TOOLCHAIN_DIR",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

_COMPILE_ACTIONS = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
]

# Bazel selects the link action based on the target type:
#   cpp_link_executable             - cc_binary / cc_test
#   cpp_link_dynamic_library        - cc_binary(linkshared = True)
#   cpp_link_nodeps_dynamic_library - .so Bazel creates for a cc_library
_EXECUTABLE_LINK_ACTIONS = [
    ACTION_NAMES.cpp_link_executable,
]
_DYNAMIC_LIBRARY_LINK_ACTIONS = [
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]
_ALL_LINK_ACTIONS = _EXECUTABLE_LINK_ACTIONS + _DYNAMIC_LIBRARY_LINK_ACTIONS

def _aarch64_android_toolchain_config_impl(ctx):
    # The toolchain is only analyzed when building for Android, so a missing
    # NDK does not affect host builds.
    if not ANDROID_NDK_AVAILABLE:
        fail(ANDROID_NDK_ERROR)

    target = "aarch64-linux-android" + ANDROID_API_LEVEL
    bin_dir = ANDROID_NDK_TOOLCHAIN_DIR + "/bin"
    sysroot_lib_dir = (ANDROID_NDK_TOOLCHAIN_DIR +
                       "/sysroot/usr/lib/aarch64-linux-android/" +
                       ANDROID_API_LEVEL)

    clang = bin_dir + "/" + target + "-clang"
    tool_paths = [
        tool_path(name = "cpp", path = clang),
        tool_path(name = "gcc", path = clang),
        tool_path(name = "ld", path = bin_dir + "/ld.lld"),
    ] + [
        tool_path(name = tool, path = bin_dir + "/llvm-" + tool)
        for tool in ["ar", "nm", "objcopy", "objdump", "strip"]
    ]

    arch_flags = feature(
        name = "aarch64_arch_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = _COMPILE_ACTIONS,
                flag_groups = [flag_group(flags = ["-march=" + ctx.attr.march])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ] + _ALL_LINK_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "--target=" + target,
                    "-march=" + ctx.attr.march,
                ])],
            ),
        ],
    )

    compile_flags = feature(
        name = "android_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = _COMPILE_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "-fPIC",
                    "-DANDROID",
                ])],
            ),
        ],
    )

    # Enabled by "bazel build -c opt ...".
    opt = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = _COMPILE_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "-O2",
                    "-DNDEBUG",
                ])],
            ),
        ],
    )

    dbg = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = _COMPILE_ACTIONS,
                flag_groups = [flag_group(flags = ["-g", "-O0"])],
            ),
        ],
    )

    linker_flags = feature(
        name = "android_linker_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = _ALL_LINK_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "-Wl,-z,nocopyreloc",
                    "-Wl,-rpath-link=" + sysroot_lib_dir,
                    "-nostdlib",
                    "-Bdynamic",
                    "-L" + sysroot_lib_dir,
                    "-lc++",
                    "-lm",
                    "-lc",
                    "-ldl",
                    "-llog",
                    "-lunwind",
                    "-L" + ANDROID_NDK_LIB_CLANG_DIR + "/lib/linux",
                    "-lclang_rt.builtins-aarch64-android",
                ])],
            ),
            flag_set(
                actions = _EXECUTABLE_LINK_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "-fPIE",
                    "-pie",
                    "-Wl,-dynamic-linker,/system/bin/linker64",
                    # Allows deploying libc++_shared.so in ../lib.
                    "-Wl,-rpath,$ORIGIN/../lib",
                    sysroot_lib_dir + "/crtbegin_dynamic.o",
                    sysroot_lib_dir + "/crtend_android.o",
                ])],
            ),
            flag_set(
                actions = _DYNAMIC_LIBRARY_LINK_ACTIONS,
                flag_groups = [flag_group(flags = [
                    "-fPIC",
                    "-Wl,-rpath,$ORIGIN/../lib",
                    "-Wl,-unresolved-symbols=ignore-in-shared-libs",
                    sysroot_lib_dir + "/crtbegin_so.o",
                    sysroot_lib_dir + "/crtend_so.o",
                ])],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "aarch64-android",
        host_system_name = "local",
        target_system_name = target,
        target_cpu = "aarch64",
        target_libc = "bionic",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        cxx_builtin_include_directories = [
            ANDROID_NDK_LIB_CLANG_DIR + "/include",
            ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include/c++/v1",
            ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include/aarch64-linux-android",
            ANDROID_NDK_TOOLCHAIN_DIR + "/sysroot/usr/include",
        ],
        features = [
            arch_flags,
            compile_flags,
            opt,
            dbg,
            linker_flags,
        ],
    )

aarch64_android_toolchain_config = rule(
    implementation = _aarch64_android_toolchain_config_impl,
    attrs = {
        "march": attr.string(
            default = "armv8-a",
            doc = "aarch64 architecture to target (e.g., armv8-a).",
        ),
    },
    provides = [CcToolchainConfigInfo],
)
