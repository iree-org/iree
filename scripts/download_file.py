#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Downloads a file from the web and decompresses it if necessary."""

import argparse
import gzip
import os
import requests


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
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

  response = requests.get(args.source_url)
  if response.status_code != 200:
    raise requests.RequestException(
        f"Failed to download file with status code {response.status_code}")

  data = response.content
  if args.source_url.endswith(".gz"):
    data = gzip.decompress(data)

  with open(args.output, "wb") as f:
    f.write(data)


if __name__ == "__main__":
  main(parse_arguments())
