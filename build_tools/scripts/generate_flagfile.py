#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a multiline flagfile.

This tool is added due to CMake's incapabilities on generating files with
multiple lines. CMake's configure_file doesn't work in our case as it can't be
triggered from a target.
"""

import argparse


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      type=str,
                      required=True,
                      help="output file to write to")
  parser.add_argument("flags", nargs="*", help="list of flags")
  return parser.parse_args()


def main(args):
  with open(args.output, "w") as f:
    f.write("\n".join(args.flags) + "\n")


if __name__ == "__main__":
  main(parse_arguments())
