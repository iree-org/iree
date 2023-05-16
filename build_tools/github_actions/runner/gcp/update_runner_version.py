#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Updates the GitHub Actions runner version for new runner VM templates to the
# latest. A hacky script automating a manual thing. Takes no args, has no
# options. Patches welcome.
#
# Usage:
#   ./update_runner_version.py

import fileinput
import hashlib
import json
import re
import string
import subprocess
import sys
import urllib

import requests

# This is using the old printf-style string formatting because we're creating
# lines that have Bash substitutions using braces
VERSION_LINE_FORMAT_STRING = 'GITHUB_RUNNER_VERSION="${GITHUB_RUNNER_VERSION:-%s}"'
DIGEST_LINE_FORMAT_STRING = 'GITHUB_RUNNER_ARCHIVE_DIGEST="${GITHUB_RUNNER_ARCHIVE_DIGEST:-%s}"'

DIGEST_SEARCH_PATTERN = r".*\bBEGIN.SHA linux-x64\b.*\b([a-fA-F0-9]{64})\b.*END.SHA linux-x64\b.*"

RUNNER_ARCHIVE_TEMPLATE = string.Template(
    "actions-runner-linux-x64-${version}.tar.gz")
ASSET_URL_TEMPLATE = string.Template(
    "https://github.com/actions/runner/releases/download/v${version}/${archive}"
)

# I think it's actually simpler to have this hardcoded than to have the script
# introspect on its own source file location.
TARGET_SCRIPT = "build_tools/github_actions/runner/gcp/create_templates.sh"


def error(*msg):
  print(*msg, file=sys.stderr)
  sys.exit(1)


if __name__ == "__main__":
  release = json.loads(
      subprocess.run(["gh", "api", "/repos/actions/runner/releases?per_page=1"],
                     check=True,
                     text=True,
                     stdout=subprocess.PIPE).stdout.strip())[0]

  if not release["tag_name"].startswith("v"):
    error(
        f"ERROR: Release tag name '{release.tag_name}' does not start with 'v' as expected"
    )

  version = release["tag_name"][1:]
  digest = None

  sha_pattern = re.compile(DIGEST_SEARCH_PATTERN)

  matches = [
      m for m in (sha_pattern.fullmatch(l)
                  for l in release["body"].splitlines()) if m
  ]

  if not matches:
    error(
        f"ERROR: No lines match digest search regex: '{DIGEST_SEARCH_PATTERN}'")

  if len(matches) > 1:
    error(f"ERROR: Multiple lines match digest search regex:", matches)

  digest = matches[0].group(1)

  archive = RUNNER_ARCHIVE_TEMPLATE.substitute(version=version)
  asset_url = ASSET_URL_TEMPLATE.substitute(version=version, archive=archive)

  # With Python 3.11 we could use hashlib.file_digest
  hash = hashlib.sha256()
  with urllib.request.urlopen(asset_url) as f:
    hash.update(f.read())

  actual_digest = hash.hexdigest()

  if digest != actual_digest:
    error(f"Digest extracted from release notes ('{digest}') does not match"
          f" digest obtained from fetching '{asset_url}' ('{actual_digest}')")

  for line in fileinput.input(files=[TARGET_SCRIPT], inplace=True):
    if line.startswith("GITHUB_RUNNER_VERSION"):
      print(VERSION_LINE_FORMAT_STRING % (version,))
    elif line.startswith("GITHUB_RUNNER_ARCHIVE_DIGEST"):
      print(DIGEST_LINE_FORMAT_STRING % (digest,))
    else:
      print(line, end="")

  print(f"Successfully updated {TARGET_SCRIPT}")
