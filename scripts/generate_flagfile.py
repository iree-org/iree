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
  parser.add_argument("--module_file",
                      type=str,
                      required=True,
                      metavar="<module-file>",
                      help="The name of the module file")
  parser.add_argument("--driver",
                      type=str,
                      required=True,
                      metavar="<driver>",
                      help="The name of the IREE driver")
  parser.add_argument("--entry_function",
                      type=str,
                      required=True,
                      metavar="<entry-function>",
                      help="The name of the entry function")
  parser.add_argument("--function_inputs",
                      type=str,
                      required=True,
                      metavar="<function-inputs>",
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
      f"--driver={args.driver}", f"--module_file={args.module_file}",
      f"--entry_function={args.entry_function}"
  ]
  lines.extend([
      ("--function_input=" + e) for e in args.function_inputs.split(",")
  ])
  lines.extend(args.additional_args.split(";"))
  content = "\n".join(lines) + "\n"

  with open(args.output, "w") as f:
    f.writelines(content)


if __name__ == "__main__":
  main(parse_arguments())
