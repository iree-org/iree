#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts takes a file like
# 'iree_base_runtime-2.9.0rc20241107-cp311-cp311-manylinux_2_28_x86_64.whl'
# with embedded version '2.9.0rc20241107' as input and then drops the
# 'rcYYYYMMDD' suffix from both the embedded version and file name.
#
# Typical usage:
#   pip install -r pypi_deploy_requirements.txt
#   ./promote_whl_from_rc_to_final.py /path/to/file.whl --delete-old-wheel

import argparse
from change_wheel_version import change_wheel_version
from packaging.version import Version
from pathlib import Path
from pkginfo import Wheel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to the input .whl file to promote",
        type=Path,
    )
    parser.add_argument(
        "--delete-old-wheel",
        help="Deletes the original wheel after successfully promoting it",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main(args):
    original_wheel_path = args.input_file
    print(f"Promoting whl from rc to final: '{original_wheel_path}'")

    original_wheel = Wheel(original_wheel_path)
    original_version = Version(original_wheel.version)
    base_version = original_version.base_version
    print(
        f"  Original wheel version is '{original_version}' with base '{base_version}'"
    )

    if str(base_version) == str(original_version):
        print("  Version is already a release version, skipping")
        return

    print(f"  Changing to base version: '{base_version}'")
    new_wheel_path = change_wheel_version(original_wheel_path, str(base_version), None)
    print(f"  New wheel path is '{new_wheel_path}'")

    new_wheel = Wheel(new_wheel_path)
    new_version = Version(new_wheel.version)
    print(f"  New wheel version is '{new_version}'")

    if args.delete_old_wheel:
        print("  Deleting original wheel")
        original_wheel_path.unlink()


if __name__ == "__main__":
    main(parse_arguments())
