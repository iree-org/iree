#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from a
# `version_info.json` and writes the corresponding
# `X.Y.ZrcYYYYMMDD` version identifier to `version_rc_info.json`.

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
import subprocess

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--write-json", action="store_true")

release_type = parser.add_mutually_exclusive_group()
release_type.add_argument("-stable", "--stable-release", action="store_true")  # default
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")
release_type.add_argument("--custom-string", action="store", type=str)

args = parser.parse_args()

if not (
    args.stable_release
    or args.nightly_release
    or args.development_release
    or args.custom_string
):
    parser.print_usage(sys.stderr)
    sys.stderr.write("error: A release type or custom string is required\n")
    sys.exit(1)

VERSION_FILE = args.path / "version.json"
VERSION_FILE_LOCAL = args.path / "version_local.json"


def load_version_info():
    with open(VERSION_FILE, "rt") as f:
        return json.load(f)


def write_version_info():
    with open(VERSION_FILE_LOCAL, "w") as f:
        json.dump(version_local, f, indent=2)
        f.write("\n")


version_info = load_version_info()

PACKAGE_VERSION = version_info.get("package-version")
CURRENT_VERSION = Version(PACKAGE_VERSION).base_version

if args.nightly_release:
    CURRENT_VERSION += "rc" + datetime.today().strftime("%Y%m%d")

if args.development_release:
    CURRENT_VERSION += (
        ".dev+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )

if args.custom_string:
    CURRENT_VERSION += args.custom_string

if args.write_json:
    version_local = {"package-version": CURRENT_VERSION}
    write_version_info()

print(CURRENT_VERSION)
