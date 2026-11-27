#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from the
# `version.json` files of the compiler, runtime and core Python packages
# and computes the shared version: the highest of them.
#
# Usage:
#   ./compute_common_version.py --stable-release

import argparse
from pathlib import Path
import json
from datetime import datetime
import subprocess

from packaging.version import Version


parser = argparse.ArgumentParser()

release_type = parser.add_mutually_exclusive_group(required=True)
release_type.add_argument("-stable", "--stable-release", action="store_true")
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")
release_type.add_argument("--version-suffix", action="store", type=str)

args = parser.parse_args()

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_COMPILER_PATH = REPO_ROOT / "compiler/version.json"
VERSION_FILE_RUNTIME_PATH = REPO_ROOT / "runtime/version.json"
VERSION_FILE_CORE_PATH = REPO_ROOT / "core/python/version.json"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


compiler_version = load_version_info(VERSION_FILE_COMPILER_PATH)
compiler_package_version = compiler_version.get("package-version")
compiler_base_version = Version(compiler_package_version).base_version

runtime_version = load_version_info(VERSION_FILE_RUNTIME_PATH)
runtime_package_version = runtime_version.get("package-version")
runtime_base_version = Version(runtime_package_version).base_version

core_version = load_version_info(VERSION_FILE_CORE_PATH)
core_package_version = core_version.get("package-version")
core_base_version = Version(core_package_version).base_version

# Compare as versions, not as strings: 3.10.0 is higher than 3.9.0.
common_version = max(
    compiler_base_version, runtime_base_version, core_base_version, key=Version
)

if args.nightly_release:
    common_version += "rc" + datetime.today().strftime("%Y%m%d")
elif args.development_release:
    common_version += (
        ".dev0+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
elif args.version_suffix:
    common_version += args.version_suffix

print(common_version)
