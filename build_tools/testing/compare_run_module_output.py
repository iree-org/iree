#!/usr/bin/env python
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare iree-run-module dumped output

This tool is used to check the consistency between two iree-run-module output
dump. It can be used to check the cross compile consistency.
"""

import argparse
import re
import sys
import numpy as np


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Compare iree-run-module outputs")
  parser.add_argument("--tol",
                      required=True,
                      type=str,
                      help="consistency tolerance")
  parser.add_argument('files', type=str, nargs='*', help='Files to be compared')
  args = parser.parse_args()
  return args


def main(args):
  data = None
  if len(args.files) != 2:
    sys.exit("Need to provide 2 input files")

  for file in args.files:
    with open(file, "r", encoding="utf-8") as f:
      p = f.read()
    # Read only numbers
    line_list = re.findall(r"[\-*\d+\.*\d*]+", p)
    # Remove unprinted numbers (elements > print_max_element_count)
    line_list = list(filter(lambda a: a != "...", line_list))
    a = np.array(line_list, dtype=float)
    if data is None:
      data = a
    else:
      data = np.vstack((data, a))

  if not np.isclose(data[0,], data[1,], atol=float(args.tol)).all():
    error = np.abs(data[0,] - data[1,])
    max_error = np.max(error)
    max_error_idx = np.argmax(error)
    sys.exit(f"Max error: {max_error} at {max_error_idx} exceeds {args.tol}")


if __name__ == "__main__":
  main(parse_arguments())
