#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Downloads a file from the web and decompresses it if necessary. NEVER Use
  this tool to download from untrusted sources, it doesn't unpack the file
  safely.
"""

import argparse
import gzip
import os
import shutil
import tarfile
import urllib.request


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Downloads a file from the web "
      "and decompresses it if necessary. NEVER Use this tool to download from "
      "untrusted sources, it doesn't unpack the file safely.")
  parser.add_argument("source_url",
                      type=str,
                      metavar="<source-url>",
                      help="Source URL to download")
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      metavar="<output-file>",
                      help="Output file path")
  return parser.parse_args()


def main(args):
  output_dir = os.path.dirname(args.output)

  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  # Open the URL and get the file-like streaming object.
  with urllib.request.urlopen(args.source_url) as response:
    if response.status != 200:
      raise RuntimeError(
          f"Failed to download file with status {response.status} {response.msg}"
      )

    if args.source_url.endswith(".tar.gz"):
      # Open tar.gz in the streaming mode.
      with tarfile.open(fileobj=response, mode="r|*") as tar_file:
        if os.path.exists(args.output):
          shutil.rmtree(args.output)
        os.makedirs(args.output)
        tar_file.extractall(args.output)

    elif args.source_url.endswith(".gz"):
      # Open gzip from a file-like object, which will be in the streaming mode.
      with gzip.open(filename=response, mode="rb") as input_file:
        with open(args.output, "wb") as output_file:
          shutil.copyfileobj(input_file, output_file)

    else:
      with open(args.output, "wb") as output_file:
        # Streaming copy.
        shutil.copyfileobj(response, output_file)


if __name__ == "__main__":
  main(parse_arguments())
