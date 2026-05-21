# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU device binary target selection."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")
load(
    ":target_map.bzl",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_DEFAULT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGETS",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGET_CODE_OBJECTS",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILIES",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILY_NAMES",
)

_BUILD_MODE_PREBUILT = "prebuilt"
_BUILD_MODE_SOURCE = "source"
_VALID_BUILD_MODES = [
    _BUILD_MODE_PREBUILT,
    _BUILD_MODE_SOURCE,
]

_PREBUILT_CODE_OBJECT_TARGETS = {
    "gfx10-1-generic": True,
    "gfx10-3-generic": True,
    "gfx11-generic": True,
    "gfx12-generic": True,
    "gfx9-4-generic": True,
    "gfx9-generic": True,
    "gfx90a": True,
}

# Keep intentionally unbuildable helper targets out of //... CI enumeration.
_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)

def _valid_selectors():
    selectors = []
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGETS)
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS)
    _append_unique(selectors, IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILY_NAMES)
    return selectors

def iree_hal_amdgpu_expand_device_binary_targets(targets):
    expanded_targets = []
    for target in targets:
        if target in IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS:
            _append_unique(expanded_targets, [target])
        elif target in IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGETS:
            _append_unique(
                expanded_targets,
                [IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGET_CODE_OBJECTS[target]],
            )
        elif target in IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILIES:
            for exact_target in IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILIES[target]:
                _append_unique(
                    expanded_targets,
                    [IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGET_CODE_OBJECTS[exact_target]],
                )
        else:
            fail("Unknown AMDGPU device binary target or family '%s'. Available: %s" % (
                target,
                ", ".join(_valid_selectors()),
            ))
    return expanded_targets

def _target_label_fragment(target):
    return target.replace("-", "_").replace(".", "_")

def _selectors_for_code_object_target(code_object_target):
    selectors = [code_object_target]
    for exact_target in IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGETS:
        if IREE_HAL_AMDGPU_DEVICE_BINARY_EXACT_TARGET_CODE_OBJECTS[exact_target] == code_object_target:
            _append_unique(selectors, [exact_target])
    for family in IREE_HAL_AMDGPU_DEVICE_BINARY_TARGET_FAMILY_NAMES:
        if code_object_target in iree_hal_amdgpu_expand_device_binary_targets([family]):
            _append_unique(selectors, [family])
    return selectors

def _device_binary_targets_flag_impl(ctx):
    valid_selectors = _valid_selectors()
    invalid_selectors = [
        selector
        for selector in ctx.build_setting_value
        if selector not in valid_selectors
    ]
    if invalid_selectors:
        fail("Unknown AMDGPU device binary target selector(s) [{}]. Available: {}".format(
            ", ".join(invalid_selectors),
            ", ".join(valid_selectors),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_device_binary_targets_flag = rule(
    implementation = _device_binary_targets_flag_impl,
    build_setting = config.string_list(flag = True),
)

def _device_binary_build_mode_flag_impl(ctx):
    if ctx.build_setting_value not in _VALID_BUILD_MODES:
        fail("Unknown AMDGPU device binary build mode '{}'. Available: {}".format(
            ctx.build_setting_value,
            ", ".join(_VALID_BUILD_MODES),
        ))
    return BuildSettingInfo(value = ctx.build_setting_value)

_device_binary_build_mode_flag = rule(
    implementation = _device_binary_build_mode_flag_impl,
    build_setting = config.string(flag = True),
)

def iree_hal_amdgpu_device_binaries(
        name,
        target_selections = None,
        target = "amdgcn-amd-amdhsa"):
    if target_selections == None:
        target_selections = IREE_HAL_AMDGPU_DEVICE_BINARY_DEFAULT_TARGETS

    _device_binary_build_mode_flag(
        name = "build_mode",
        build_setting_default = _BUILD_MODE_PREBUILT,
    )

    native.config_setting(
        name = "build_mode_prebuilt",
        flag_values = {
            ":build_mode": _BUILD_MODE_PREBUILT,
        },
    )

    native.config_setting(
        name = "build_mode_source",
        flag_values = {
            ":build_mode": _BUILD_MODE_SOURCE,
        },
    )

    _device_binary_targets_flag(
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
    for code_object_target in IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS:
        binary_name = "%s--%s" % (target, code_object_target)
        selects.config_setting_group(
            name = "%s_requested" % (_target_label_fragment(code_object_target),),
            match_any = [
                ":%s_selected" % (_target_label_fragment(selector),)
                for selector in _selectors_for_code_object_target(code_object_target)
            ],
        )
        selects.config_setting_group(
            name = "%s_requested_prebuilt" % (_target_label_fragment(code_object_target),),
            match_all = [
                ":%s_requested" % (_target_label_fragment(code_object_target),),
                ":build_mode_prebuilt",
            ],
        )
        selects.config_setting_group(
            name = "%s_requested_source" % (_target_label_fragment(code_object_target),),
            match_all = [
                ":%s_requested" % (_target_label_fragment(code_object_target),),
                ":build_mode_source",
            ],
        )
        if code_object_target in _PREBUILT_CODE_OBJECT_TARGETS:
            prebuilt_label = "//runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt:%s.so" % (binary_name,)
        else:
            missing_prebuilt_name = "missing_prebuilt_%s" % (_target_label_fragment(code_object_target),)
            native.filegroup(
                name = missing_prebuilt_name,
                target_compatible_with = _INCOMPATIBLE_TARGET,
            )
            prebuilt_label = ":%s" % (missing_prebuilt_name,)
        source_label = "//runtime/src/iree/hal/drivers/amdgpu/device/binaries/source:%s.so" % (binary_name,)
        binary_srcs += select({
            ":%s_requested_prebuilt" % (_target_label_fragment(code_object_target),): [prebuilt_label],
            ":%s_requested_source" % (_target_label_fragment(code_object_target),): [source_label],
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
