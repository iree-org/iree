#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Downloads a file from the web and untars it if necessary."""

import argparse
import os
import requests
import tarfile


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
                      metavar="<output-directory>",
                      help="Output directory to contain the file")
  return parser.parse_args()


def main(args):
  name = args.source_url.split("/")[-1]

  if not os.path.isdir(args.output):
    os.makedirs(args.output)
  output_file = os.path.join(args.output, name)

  response = requests.get(args.source_url)
  if response.status_code != 200:
    raise requests.RequestException(
        f"Failed to download file with status code {response.status_code}")

  with open(output_file, "wb") as f:
    f.write(response.content)

  if name.endswith("tar.gz") or name.endswith("tgz"):
    with tarfile.open(output_file, "r") as f:
      f.extractall(args.output)


if __name__ == "__main__":
  main(parse_arguments())
