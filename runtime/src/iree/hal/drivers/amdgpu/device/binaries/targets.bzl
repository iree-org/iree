# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU device library target selection."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//build_tools/bazel:iree_amdgpu_binary.bzl", "iree_amdgpu_binary")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")
load(
    ":target_map.bzl",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_CODE_OBJECT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_DEFAULT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGET_CODE_OBJECTS",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILIES",
    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILY_NAMES",
)

def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)

def _valid_selectors():
    selectors = []
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS)
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_LIBRARY_CODE_OBJECT_TARGETS)
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILY_NAMES)
    return selectors

def iree_hal_amdgpu_expand_device_library_targets(targets):
    expanded_targets = []
    for target in targets:
        if target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_CODE_OBJECT_TARGETS:
            _append_unique(expanded_targets, [target])
        elif target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS:
            _append_unique(
                expanded_targets,
                [IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGET_CODE_OBJECTS[target]],
            )
        elif target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILIES:
            for exact_target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILIES[target]:
                _append_unique(
                    expanded_targets,
                    [IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGET_CODE_OBJECTS[exact_target]],
                )
        else:
            fail("Unknown AMDGPU device library target or family '%s'. Available: %s" % (
                target,
                ", ".join(_valid_selectors()),
            ))
    return expanded_targets

def _target_label_fragment(target):
    return target.replace("-", "_").replace(".", "_")

def _selectors_for_code_object_target(code_object_target):
    selectors = [code_object_target]
    for exact_target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS:
        if IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGET_CODE_OBJECTS[exact_target] == code_object_target:
            _append_unique(selectors, [exact_target])
    for family in IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILY_NAMES:
        if code_object_target in iree_hal_amdgpu_expand_device_library_targets([family]):
            _append_unique(selectors, [family])
    return selectors

def _device_library_targets_flag_impl(ctx):
    valid_selectors = _valid_selectors()
    invalid_selectors = [
        selector
        for selector in ctx.build_setting_value
        if selector not in valid_selectors
    ]
    if invalid_selectors:
        fail("Unknown AMDGPU device library target selector(s) [{}]. Available: {}".format(
            ", ".join(invalid_selectors),
            ", ".join(valid_selectors),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_device_library_targets_flag = rule(
    implementation = _device_library_targets_flag_impl,
    build_setting = config.string_list(flag = True),
)

def iree_hal_amdgpu_device_binaries(
        name,
        srcs,
        internal_hdrs,
        target_selections = None,
        target = "amdgcn-amd-amdhsa"):
    if target_selections == None:
        target_selections = IREE_HAL_AMDGPU_DEVICE_LIBRARY_DEFAULT_TARGETS

    _device_library_targets_flag(
        name = "targets",
        build_setting_default = target_selections,
    )

    for selector in _valid_selectors():
        native.config_setting(
            name = "%s_selected" % (_target_label_fragment(selector),),
            flag_values = {
                ":targets": selector,
            },
        )

    binary_srcs = []
    for code_object_target in IREE_HAL_AMDGPU_DEVICE_LIBRARY_CODE_OBJECT_TARGETS:
        binary_name = "%s--%s" % (target, code_object_target)
        iree_amdgpu_binary(
            name = binary_name,
            srcs = srcs,
            arch = code_object_target,
            internal_hdrs = internal_hdrs,
            target = target,
        )
        selects.config_setting_group(
            name = "%s_requested" % (_target_label_fragment(code_object_target),),
            match_any = [
                ":%s_selected" % (_target_label_fragment(selector),)
                for selector in _selectors_for_code_object_target(code_object_target)
            ],
        )
        binary_srcs += select({
            ":%s_requested" % (_target_label_fragment(code_object_target),): [":%s.so" % (binary_name,)],
            "//conditions:default": [],
        })
    iree_c_embed_data(
        name = name,
        srcs = binary_srcs,
        c_file_output = "toc.c",
        flatten = True,
        h_file_output = "toc.h",
        identifier = "iree_hal_amdgpu_device_binaries",
    )
