#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a flagfile containing command-line options for artifact compilation."""

import argparse


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      metavar="<output-file>",
                      help="Output file to write to")
  parser.add_argument("cloptions",
                      type=str,
                      nargs="+",
                      metavar="<command-line-option>",
                      help="Command-line option used to generate artifact")
  return parser.parse_args()


def main(args):
  content = "\n".join(args.cloptions) + "\n"

  with open(args.output, "w") as f:
    f.writelines(content)


if __name__ == "__main__":
  main(parse_arguments())
