# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for compiler plugin registration in Bazel builds.

This provides the Bazel equivalent of CMake's iree_compiler_register_plugin().
CMake uses global properties; Bazel uses select() with config_settings.
"""

load("@bazel_skylib//rules:common_settings.bzl", "string_list_flag")

def _plugin_inc_select(plugin_id):
    """Returns a select() for conditionally including a plugin's .inc file."""
    return select({
        ":" + plugin_id + "_enabled": [":" + plugin_id + "_plugin_id"],
        "//conditions:default": [],
    })

def _plugin_dep_select(plugin_id, registration_target):
    """Returns a select() for conditionally including a plugin's registration dep."""
    return select({
        ":" + plugin_id + "_enabled": [registration_target],
        "//conditions:default": [],
    })

def iree_compiler_plugins(name, plugins, flag_name = "enabled_plugins", default_plugins = None):
    """Registers compiler plugins and creates aggregation targets.

    This macro creates:
    - string_list_flag: {flag_name} for selecting which plugins to enable
    - Per-plugin config_setting: {plugin_id}_enabled
    - Per-plugin genrule: {plugin_id}_plugin_id (outputs {plugin_id}.inc)
    - Aggregate filegroup: {name}_incs (all enabled .inc files)
    - Aggregate cc_library: {name}_deps (all enabled registration deps)

    Args:
        name: Base name for aggregate targets.
        plugins: Dict mapping plugin_id -> registration_target.
        flag_name: Name for the string_list_flag (default: "enabled_plugins").
        default_plugins: List of plugin IDs enabled by default. If None, all plugins
            are enabled by default.
    """

    # Determine default plugins
    if default_plugins == None:
        default_plugins = plugins.keys()

    # Create the flag for selecting plugins
    string_list_flag(
        name = flag_name,
        build_setting_default = list(default_plugins),
        visibility = ["//visibility:public"],
    )

    # Create per-plugin config_settings and genrules
    for plugin_id in plugins.keys():
        native.config_setting(
            name = plugin_id + "_enabled",
            flag_values = {":" + flag_name: plugin_id},
        )
        native.genrule(
            name = plugin_id + "_plugin_id",
            outs = [plugin_id + ".inc"],
            cmd = "echo 'HANDLE_PLUGIN_ID({})' > $@".format(plugin_id),
            visibility = ["//visibility:private"],
        )

    # Create aggregate filegroup for .inc files
    inc_srcs = []
    for plugin_id in plugins.keys():
        inc_srcs = inc_srcs + _plugin_inc_select(plugin_id)

    native.filegroup(
        name = name + "_incs",
        srcs = inc_srcs,
        visibility = ["//visibility:public"],
    )

    # Create aggregate cc_library that depends on all enabled plugins
    plugin_deps = []
    for plugin_id, registration_target in plugins.items():
        plugin_deps = plugin_deps + _plugin_dep_select(plugin_id, registration_target)

    native.cc_library(
        name = name + "_deps",
        deps = plugin_deps,
        visibility = ["//visibility:public"],
    )
