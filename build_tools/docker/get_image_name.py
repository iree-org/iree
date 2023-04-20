#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command line tool to get the fully qualified image name given short name.

Syntax:
  ./build_tools/docker/get_image_name.py {short_name}

Where {short_name} is the last name component of an image in prod_digests.txt
(i.e. "base", "nvidia", etc).

This is used both in tree and out of tree to get a image name and current
version without adding fully referencing sha256 hashes, etc.
"""

from pathlib import Path
import sys


def find_image_by_name(img_name):
  this_dir = Path(__file__).resolve().parent

  with open(this_dir / "prod_digests.txt", "rt") as f:
    for line in f.readlines():
      line = line.strip()
      if line.startswith(f"gcr.io/iree-oss/{img_name}@"):
        return line
    else:
      raise ValueError(
          f"ERROR: Image name {img_name} not found in prod_digests.txt")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("ERROR: Expected image short name", file=sys.stderr)
    sys.exit(1)
  short_name = sys.argv[1]
  image_name = find_image_by_name(short_name)
  print(image_name)
