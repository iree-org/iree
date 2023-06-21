#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command line tool to get the fully qualified image name given short name.

Syntax:
  ./build_tools/docker/get_image_name.py {short_name}

Where {short_name} is the last name component of an image in image_graph.json
(i.e. "base", "nvidia", etc).

This is used both in tree and out of tree to get a image name and current
version without adding fully referencing sha256 hashes, etc.
"""

from pathlib import Path
import sys

import utils


def find_image_by_name(img_name):
    this_dir = Path(__file__).resolve().parent

    image_graph = utils.load_image_graph(this_dir / "image_graph.json")
    image_info = image_graph.get(img_name)
    if image_info is None:
        raise ValueError(f"ERROR: Image name {img_name} not found in image_graph.json")

    return image_info.url


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Expected image short name", file=sys.stderr)
        sys.exit(1)
    short_name = sys.argv[1]
    image_name = find_image_by_name(short_name)
    print(image_name)
