#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from the
# compiler's and runtime's `version.json` files and computes the
# shared version.

import argparse
from pathlib import Path
import json
from datetime import datetime
import subprocess

from packaging.version import Version


parser = argparse.ArgumentParser()
release_type = parser.add_mutually_exclusive_group()
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")
release_type.add_argument("--custom-string", action="store", type=str)
args = parser.parse_args()

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_COMPILER = REPO_ROOT / "compiler/version.json"
VERSION_FILE_RUNTIME = REPO_ROOT / "runtime/version.json"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


compiler_version = load_version_info(VERSION_FILE_COMPILER)
COMPILER_PACKAGE_VERSION = compiler_version.get("package-version")
COMPILER_BASE_VERSION = Version(COMPILER_PACKAGE_VERSION).base_version

runtime_version = load_version_info(VERSION_FILE_RUNTIME)
RUNTIME_PACKAGE_VERSION = runtime_version.get("package-version")
RUNTIME_BASE_VERSION = Version(RUNTIME_PACKAGE_VERSION).base_version

if RUNTIME_BASE_VERSION > COMPILER_BASE_VERSION:
    CURRENT_VERSION = RUNTIME_BASE_VERSION
else:
    CURRENT_VERSION = COMPILER_BASE_VERSION

if args.nightly_release:
    CURRENT_VERSION += "rc" + datetime.today().strftime("%Y%m%d")

if args.development_release:
    CURRENT_VERSION += (
        ".dev+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )

if args.custom_string:
    CURRENT_VERSION += args.custom_string

print(CURRENT_VERSION)
