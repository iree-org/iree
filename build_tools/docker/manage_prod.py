#!/usr/bin/env python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Uses prod_digests.txt to update GCR's :prod tags.

Usage:
  Pull all images that should have :prod tags, tag them with :prod and push
  them to GCR. This will make sure that you are at upstream head on the main
  branch before pushing:
    python3 build_tools/docker/manage_prod.py
"""

import os
import utils

if __name__ == "__main__":
  # Ensure the user has the correct authorization if they try to push to GCR.
  utils.check_gcloud_auth()

  # Only allow the :prod tag to be pushed from the version of
  # `prod_digests.txt` at upstream HEAD on the main branch.
  utils.run_command(
      [os.path.normpath("build_tools/scripts/git/git_update.sh"), "main"])

  with open(utils.PROD_DIGESTS_PATH, "r") as f:
    images_with_digests = [line.strip() for line in f.readlines()]

  for image_with_digest in images_with_digests:
    image_url, _ = image_with_digest.split("@")
    prod_image_url = f"{image_url}:prod"

    utils.run_command(["docker", "pull", image_with_digest])
    utils.run_command(["docker", "tag", image_with_digest, prod_image_url])
    utils.run_command(["docker", "push", prod_image_url])
