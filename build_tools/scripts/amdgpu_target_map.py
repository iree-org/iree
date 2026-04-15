#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates shared AMDGPU device library target map fragments.

The map in this file is the source of truth for the small generated tables used
by Bazel, CMake, and the runtime device-library loader. Keep build logic in
Starlark/CMake; keep target facts here.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

DEFAULT_TARGET_SELECTIONS = ("all",)

# Each exact target must match an HSA ISA architecture suffix. Each code object
# target must be accepted by LLVM clang/lld as an AMDGPU -march value. Generic
# code object coverage follows LLVM generic processor documentation; TheRock
# family membership follows ROCm/TheRock's cmake/therock_amdgpu_targets.cmake.
EXACT_TARGET_CODE_OBJECTS = (
    ("gfx900", "gfx9-generic"),
    ("gfx90c", "gfx9-generic"),
    ("gfx906", "gfx9-generic"),
    ("gfx908", "gfx908"),
    ("gfx90a", "gfx90a"),
    ("gfx942", "gfx9-4-generic"),
    ("gfx950", "gfx9-4-generic"),
    ("gfx1010", "gfx10-1-generic"),
    ("gfx1011", "gfx10-1-generic"),
    ("gfx1012", "gfx10-1-generic"),
    ("gfx1030", "gfx10-3-generic"),
    ("gfx1031", "gfx10-3-generic"),
    ("gfx1032", "gfx10-3-generic"),
    ("gfx1033", "gfx10-3-generic"),
    ("gfx1034", "gfx10-3-generic"),
    ("gfx1035", "gfx10-3-generic"),
    ("gfx1036", "gfx10-3-generic"),
    ("gfx1100", "gfx11-generic"),
    ("gfx1101", "gfx11-generic"),
    ("gfx1102", "gfx11-generic"),
    ("gfx1103", "gfx11-generic"),
    ("gfx1150", "gfx11-generic"),
    ("gfx1151", "gfx11-generic"),
    ("gfx1152", "gfx11-generic"),
    ("gfx1153", "gfx11-generic"),
    ("gfx1200", "gfx12-generic"),
    ("gfx1201", "gfx12-generic"),
)

ALL_EXACT_TARGETS = object()

TARGET_FAMILIES = (
    ("all", ALL_EXACT_TARGETS),
    ("dcgpu-all", ("gfx908", "gfx90a", "gfx942", "gfx950")),
    (
        "dgpu-all",
        (
            "gfx900",
            "gfx906",
            "gfx1010",
            "gfx1011",
            "gfx1012",
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1034",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1200",
            "gfx1201",
        ),
    ),
    ("gfx900-dgpu", ("gfx900",)),
    ("gfx906-dgpu", ("gfx906",)),
    ("gfx908-dcgpu", ("gfx908",)),
    ("gfx90a-dcgpu", ("gfx90a",)),
    ("gfx90c-igpu", ("gfx90c",)),
    ("gfx94X-all", ("gfx942",)),
    ("gfx94X-dcgpu", ("gfx942",)),
    ("gfx950-all", ("gfx950",)),
    ("gfx950-dcgpu", ("gfx950",)),
    ("gfx101X-all", ("gfx1010", "gfx1011", "gfx1012")),
    ("gfx101X-dgpu", ("gfx1010", "gfx1011", "gfx1012")),
    (
        "gfx103X-all",
        (
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1033",
            "gfx1034",
            "gfx1035",
            "gfx1036",
        ),
    ),
    ("gfx103X-dgpu", ("gfx1030", "gfx1031", "gfx1032", "gfx1034")),
    ("gfx103X-igpu", ("gfx1033", "gfx1035", "gfx1036")),
    ("gfx110X-all", ("gfx1100", "gfx1101", "gfx1102", "gfx1103")),
    ("gfx110X-dgpu", ("gfx1100", "gfx1101", "gfx1102")),
    ("gfx110X-igpu", ("gfx1103",)),
    ("gfx115X-all", ("gfx1150", "gfx1151", "gfx1152", "gfx1153")),
    ("gfx115X-igpu", ("gfx1150", "gfx1151", "gfx1152", "gfx1153")),
    ("gfx120X-all", ("gfx1200", "gfx1201")),
    (
        "igpu-all",
        (
            "gfx90c",
            "gfx1033",
            "gfx1035",
            "gfx1036",
            "gfx1103",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
        ),
    ),
)


def find_repo_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "runtime" / "src" / "iree").exists():
            return current
        current = current.parent
    print("error: could not find IREE repository root", file=sys.stderr)
    sys.exit(1)


def append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)


def exact_targets():
    return [exact_target for exact_target, _ in EXACT_TARGET_CODE_OBJECTS]


def code_object_targets():
    values = []
    for _, code_object_target in EXACT_TARGET_CODE_OBJECTS:
        append_unique(values, [code_object_target])
    return values


def family_targets(targets):
    if targets is ALL_EXACT_TARGETS:
        return exact_targets()
    return list(targets)


def target_family_names():
    return [family for family, _ in TARGET_FAMILIES]


def validate_target_map():
    exact = exact_targets()
    if len(set(exact)) != len(exact):
        raise ValueError("duplicate exact AMDGPU targets in target map")

    families = target_family_names()
    if len(set(families)) != len(families):
        raise ValueError("duplicate AMDGPU target families in target map")

    exact_set = set(exact)
    for family, targets in TARGET_FAMILIES:
        unknown_targets = sorted(set(family_targets(targets)) - exact_set)
        if unknown_targets:
            raise ValueError(
                "target family {} references unknown exact targets: {}".format(
                    family, ", ".join(unknown_targets)
                )
            )


def generated_header(comment_prefix, output_path):
    return "\n".join(
        [
            "{} Generated by build_tools/scripts/amdgpu_target_map.py.".format(
                comment_prefix
            ),
            "{} Do not edit directly; edit the map in that script and regenerate.".format(
                comment_prefix
            ),
            "{} Output: {}".format(comment_prefix, output_path),
        ]
    )


def bzl_list(name, values):
    lines = ["{} = [".format(name)]
    lines.extend(['    "{}",'.format(value) for value in values])
    lines.append("]")
    return "\n".join(lines)


def bzl_string_dict(name, values):
    lines = ["{} = {{".format(name)]
    for key, value in values:
        lines.append('    "{}": "{}",'.format(key, value))
    lines.append("}")
    return "\n".join(lines)


def bzl_family_dict(name):
    lines = ["{} = {{".format(name)]
    for family, targets in TARGET_FAMILIES:
        values = family_targets(targets)
        if targets is ALL_EXACT_TARGETS:
            lines.append(
                '    "{}": IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS,'.format(family)
            )
        elif len(values) == 1:
            lines.append('    "{}": ["{}"],'.format(family, values[0]))
        else:
            lines.append('    "{}": ['.format(family))
            lines.extend(['        "{}",'.format(value) for value in values])
            lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def render_bzl():
    output_path = "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.bzl"
    return (
        "\n\n".join(
            [
                generated_header("#", output_path),
                bzl_list(
                    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_DEFAULT_TARGETS",
                    DEFAULT_TARGET_SELECTIONS,
                ),
                bzl_list(
                    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGETS",
                    exact_targets(),
                ),
                bzl_list(
                    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_CODE_OBJECT_TARGETS",
                    code_object_targets(),
                ),
                bzl_string_dict(
                    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_EXACT_TARGET_CODE_OBJECTS",
                    EXACT_TARGET_CODE_OBJECTS,
                ),
                bzl_list(
                    "IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILY_NAMES",
                    target_family_names(),
                ),
                bzl_family_dict("IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGET_FAMILIES"),
            ]
        )
        + "\n"
    )


def cmake_list(name, values):
    lines = ["set({}".format(name)]
    lines.extend(['  "{}"'.format(value) for value in values])
    lines.append(")")
    return "\n".join(lines)


def cmake_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def render_cmake():
    output_path = "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.cmake"
    lines = [
        generated_header("#", output_path),
        "",
        cmake_list("_IREE_HAL_AMDGPU_DEVICE_TARGETS", exact_targets()),
        "",
        cmake_list(
            "_IREE_HAL_AMDGPU_DEVICE_CODE_OBJECT_TARGETS", code_object_targets()
        ),
        "",
    ]
    for exact_target, code_object_target in EXACT_TARGET_CODE_OBJECTS:
        lines.append(
            'set(_IREE_HAL_AMDGPU_DEVICE_TARGET_CODE_OBJECT_{} "{}")'.format(
                exact_target, code_object_target
            )
        )
    lines.extend(
        [
            "",
            cmake_list(
                "_IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILIES", target_family_names()
            ),
            "",
        ]
    )
    for family, targets in TARGET_FAMILIES:
        var_name = "_IREE_HAL_AMDGPU_DEVICE_TARGET_FAMILY_{}".format(
            cmake_identifier(family)
        )
        if targets is ALL_EXACT_TARGETS:
            lines.append("set({}".format(var_name))
            lines.append("  ${_IREE_HAL_AMDGPU_DEVICE_TARGETS}")
            lines.append(")")
        else:
            lines.append(cmake_list(var_name, family_targets(targets)))
    lines.append("")
    return "\n".join(lines)


def render_inl():
    output_path = "runtime/src/iree/hal/drivers/amdgpu/device/binaries/target_map.inl"
    lines = [
        generated_header("//", output_path),
        "//",
        "// Included inside iree_hal_amdgpu_device_library_isa_mappings.",
        "",
        "// clang-format off",
    ]
    for exact_target, code_object_target in EXACT_TARGET_CODE_OBJECTS:
        lines.append(
            '{{IREE_SVL("{}"), IREE_SVL("{}")}},'.format(
                exact_target, code_object_target
            )
        )
    lines.append("")
    return "\n".join(lines)


def generated_outputs(repo_root):
    output_dir = repo_root / "runtime/src/iree/hal/drivers/amdgpu/device/binaries"
    return {
        output_dir / "target_map.bzl": render_bzl(),
        output_dir / "target_map.cmake": render_cmake(),
        output_dir / "target_map.inl": render_inl(),
    }


def check_outputs(repo_root, outputs):
    failed = False
    for path, content in outputs.items():
        if not path.exists():
            print("error: {} does not exist".format(path), file=sys.stderr)
            failed = True
            continue
        existing = path.read_text()
        if existing == content:
            continue
        rel_path = path.relative_to(repo_root)
        print("error: {} is out of date".format(rel_path), file=sys.stderr)
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(rel_path),
            tofile=str(rel_path) + " (generated)",
        )
        sys.stderr.writelines(diff)
        failed = True
    if failed:
        print(
            "Run 'python build_tools/scripts/amdgpu_target_map.py' to regenerate.",
            file=sys.stderr,
        )
        return 1
    print("AMDGPU target map generated files are up to date.")
    return 0


def write_outputs(outputs):
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print("Wrote {}".format(path))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate AMDGPU device library target map fragments."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that generated files are up to date without modifying them.",
    )
    args = parser.parse_args()

    validate_target_map()
    repo_root = find_repo_root()
    outputs = generated_outputs(repo_root)
    if args.check:
        return check_outputs(repo_root, outputs)
    return write_outputs(outputs)


if __name__ == "__main__":
    sys.exit(main())
