# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Computes the version of the iree-importer-torch package, in the same way as the
# compiler and runtime packages do. The other metadata is in pyproject.toml.
#
# Building from an sdist is not supported.

import json
import os
import subprocess
import sys

from setuptools import setup

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
IREE_SOURCE_DIR = os.path.join(SETUPPY_DIR, "..", "..", "..")

VERSION_FILE = os.path.join(SETUPPY_DIR, "version.json")
VERSION_FILE_LOCAL = os.path.join(SETUPPY_DIR, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def find_git_revision():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=IREE_SOURCE_DIR)
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"ERROR: Could not get IREE revision: {e}", file=sys.stderr)
        return ""


is_dev_build = False
try:
    version_info = load_version_info(VERSION_FILE_LOCAL)
except FileNotFoundError:
    print("version_local.json not found. Using version.json defaults")
    version_info = load_version_info(VERSION_FILE)
    is_dev_build = True

# A "package-suffix" in the version info is ignored: it tells the build
# variants of the compiler and runtime packages apart, and this package has
# none.
PACKAGE_VERSION = version_info.get("package-version")
if is_dev_build:
    PACKAGE_VERSION += f"+{find_git_revision() or '0'}"

setup(version=PACKAGE_VERSION)
