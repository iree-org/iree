#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a flagfile for iree-benchmark-module."""

import argparse
import os


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--module",
                      type=str,
                      required=True,
                      metavar="<module>",
                      help="The name of the module file")
  parser.add_argument("--device",
                      type=str,
                      required=True,
                      metavar="<device>",
                      help="The name of the HAL device")
  parser.add_argument("--function",
                      type=str,
                      required=True,
                      metavar="<function>",
                      help="The name of the entry function")
  parser.add_argument("--inputs",
                      type=str,
                      required=True,
                      metavar="<inputs>",
                      help="A list of comma-separated function inputs")
  parser.add_argument("--additional_args",
                      type=str,
                      required=True,
                      metavar="<additional-cl-args>",
                      help="Additional command-line arguments")
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      metavar="<output-file>",
                      help="Output file to write to")
  return parser.parse_args()


def main(args):
  lines = [
      f"--device={args.device}", f"--module={args.module}",
      f"--function={args.function}"
  ]
  lines.extend([("--input=" + e) for e in args.inputs.split(",")])
  lines.extend(args.additional_args.split(";"))
  content = "\n".join(lines) + "\n"

  with open(args.output, "w") as f:
    f.writelines(content)


if __name__ == "__main__":
  main(parse_arguments())
