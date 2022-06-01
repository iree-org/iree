#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a compilation flagfile for iree-compiler."""

import argparse


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      type=str,
                      required=True,
                      help="output file to write to")
  parser.add_argument("compilation_flags",
                      metavar="<compilation-flags>",
                      nargs="*",
                      help="list of compilation flags")
  return parser.parse_args()


def main(args):
  with open(args.output, "w") as f:
    f.write("\n".join(args.compilation_flags) + "\n")


if __name__ == "__main__":
  main(parse_arguments())
