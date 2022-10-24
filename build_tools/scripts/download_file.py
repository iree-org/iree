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
import urllib.error
import logging
import time

DEFAULT_MAX_TRIES = 3
RETRY_COOLDOWN_TIME = 1.0


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
  parser.add_argument("--unpack",
                      action='store_true',
                      default=False,
                      help="Unpack the downloaded file if it's an archive")
  parser.add_argument("--max-tries",
                      metavar="<max-tries>",
                      type=int,
                      default=DEFAULT_MAX_TRIES,
                      help="Number of tries before giving up")
  return parser.parse_args()


def download_and_extract(source_url: str, output: str, unpack: bool):
  # Open the URL and get the file-like streaming object.
  with urllib.request.urlopen(source_url) as response:
    if response.status != 200:
      raise RuntimeError(
          f"Failed to download file with status {response.status} {response.msg}"
      )

    if unpack:
      if source_url.endswith(".tar.gz"):
        # Open tar.gz in the streaming mode.
        with tarfile.open(fileobj=response, mode="r|*") as tar_file:
          if os.path.exists(output):
            shutil.rmtree(output)
          os.makedirs(output)
          tar_file.extractall(output)
        return
      elif source_url.endswith(".gz"):
        # Open gzip from a file-like object, which will be in the streaming mode.
        with gzip.open(filename=response, mode="rb") as input_file:
          with open(output, "wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        return

    # Fallback to download the file only.
    with open(output, "wb") as output_file:
      # Streaming copy.
      shutil.copyfileobj(response, output_file)


def main(args):
  output_dir = os.path.dirname(args.output)

  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  remaining_tries = args.max_tries
  while remaining_tries > 0:
    try:
      download_and_extract(args.source_url, args.output, args.unpack)
      break
    except (ConnectionResetError, ConnectionRefusedError,
            urllib.error.URLError):
      remaining_tries -= 1
      if remaining_tries == 0:
        raise
      else:
        logging.warning(f"Connection error, remaining {remaining_tries} tries",
                        exc_info=True)
        time.sleep(RETRY_COOLDOWN_TIME)


if __name__ == "__main__":
  main(parse_arguments())
