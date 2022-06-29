#!/usr/bin/env python
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Fetches all wheels and pypi artifacts from a github release.
# Usage:
#   GITHUB_USER=github_username:token \
#   fetch_github_release_files candidate-20220108.8 ~/tmp/wheels
#
# You can then upload to pypi via:
#   pip install twine
#   export TWINE_USERNAME=...
#   export TWINE_PASSWORD='...'
#   twine upload ~/tmp/wheels

import requests
import requests.auth
import json
import os
import sys


def main(args):
  if len(args) < 2:
    print("Syntax: fetch_github_release_files.py <tag> <dir>")
    sys.exit(1)
  tag = args[0]
  dir = args[1]
  github_user = os.getenv("GITHUB_USER")
  github_auth = None
  if github_user is not None:
    print("Using github user from GITHUB_USER env var")
    github_auth = requests.auth.HTTPBasicAuth(github_user, "")
  else:
    print("No github user set. Recommend setting GITHUB_USER=user:token")
  print("Fetching release from tag:", tag)
  release_resp = requests.get(
      f"https://api.github.com/repos/iree-org/iree/releases/tags/{tag}",
      headers={"Accept": "application/vnd.github.v3+json"},
      auth=github_auth)
  release_resp.raise_for_status()
  release_json = json.loads(release_resp.text)
  assets = release_json["assets"]
  print(f"Release contains {len(assets)} assets: ")

  os.makedirs(dir, exist_ok=True)
  for asset in assets:
    asset_name = asset["name"]
    asset_url = asset["url"]
    if not asset_name.endswith(".whl"):
      print(f"SKIP: {asset_name}")
      continue
    print(f"Downloading {asset_name} from {asset_url}")
    asset_resp = requests.get(asset_url,
                              headers={"Accept": "application/octet-stream"},
                              auth=github_auth)
    asset_resp.raise_for_status()
    dest_file = os.path.join(dir, asset_name)
    with open(dest_file, "wb") as f:
      f.write(asset_resp.content)


if __name__ == "__main__":
  main(sys.argv[1:])
