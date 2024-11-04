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

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--write-json", action="store_true")
args = parser.parse_args()

VERSION_INFO_FILE = args.path / "version_info.json"
VERSION_INFO_RC_FILE = args.path / "version_info_rc.json"


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


def write_version_info():
    with open(VERSION_INFO_RC_FILE, "w") as f:
        json.dump(version_info_rc, f, indent=2)
        f.write("\n")


version_info = load_version_info()

PACKAGE_VERSION = version_info.get("package-version")
PACKAGE_BASE_VERSION = Version(PACKAGE_VERSION).base_version
PACKAGE_RC_VERSION = PACKAGE_BASE_VERSION + "rc" + datetime.today().strftime("%Y%m%d")

if args.write_json:
    version_info_rc = {"package-version": PACKAGE_RC_VERSION}
    write_version_info()

print(PACKAGE_RC_VERSION)
