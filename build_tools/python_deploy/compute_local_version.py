#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from a
# `version.json` and writes a version identifier for a stable,
# nightly or development release, or a release with an arbitrary
# `X.Y.ZrcYYYYMMDD` version identifier to `version_local.json`.

import argparse
from pathlib import Path
import json
from datetime import datetime
import subprocess

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--write-json", action="store_true")

release_type = parser.add_mutually_exclusive_group(required=True)
release_type.add_argument("-stable", "--stable-release", action="store_true")
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")
release_type.add_argument("--version-suffix", action="store", type=str)

args = parser.parse_args()

VERSION_FILE_PATH = args.path / "version.json"
VERSION_FILE_LOCAL_PATH = args.path / "version_local.json"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_version_info(version_file, version):
    with open(version_file, "w") as f:
        json.dump({"package-version": version}, f, indent=2)
        f.write("\n")


version_info = load_version_info(VERSION_FILE_PATH)
package_version = version_info.get("package-version")
current_version = Version(package_version).base_version

if args.nightly_release:
    current_version += "rc" + datetime.today().strftime("%Y%m%d")
elif args.development_release:
    current_version += (
        ".dev0+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
elif args.version_suffix:
    current_version += args.version_suffix

if args.write_json:
    write_version_info(VERSION_FILE_LOCAL_PATH, current_version)

print(current_version)
