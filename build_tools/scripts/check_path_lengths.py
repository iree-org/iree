#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scans the IREE source tree for long path lengths, which are problematic
# on Windows: https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
#
# To check that all relative paths are shorter than the default limit:
#   python check_path_lengths.py
#
# To check that all relative paths are shorter than a custom limit:
#   python check_path_lengths.py --limit=50

import argparse
import os
import sys


def repo_relpath(repo_root, path):
  return os.path.relpath(path, repo_root).replace("\\", "/")


def parse_arguments():
  parser = argparse.ArgumentParser(description="Path length checker")
  parser.add_argument("--limit",
                      help="Path length limit (inclusive)",
                      type=int,
                      default=75)
  parser.add_argument("--include_tests",
                      help="Includes /test directories",
                      action="store_true",
                      default=False)
  args = parser.parse_args()
  return args


def main(args):
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  # Just look at the compiler directory for now.
  compiler_dir = os.path.join(repo_root, "compiler")

  directories = [repo_relpath(repo_root, x[0]) for x in os.walk(compiler_dir)]
  if not args.include_tests:
    directories = [dir for dir in directories if not dir.endswith("test")]
  sorted_directories = sorted(directories, key=len)

  if not sorted_directories:
    print("Did not find any directories")
    return

  print("*** Path length, relative path (sorted by path length) ***")

  print("Below the limit of {}:".format(args.limit))
  passed_limit = False
  number_above_limit = 0
  number_below_limit = 0

  for dir in sorted_directories:
    dir_length = len(dir)
    if dir_length > args.limit and not passed_limit:
      print("Above the limit of {}:".format(args.limit))
      passed_limit = True

    if dir_length <= args.limit:
      number_below_limit += 1
    else:
      number_above_limit += 1

    print("{:3d}, {}".format(dir_length, dir))

  # TODO(scotttodd): also scan/check file name lengths?

  print("*** Summary ***")
  print("{:3d} paths above the {} character limit".format(
      number_above_limit, args.limit))
  print("{:3d} paths below the {} character limit".format(
      number_below_limit, args.limit))

  if number_above_limit > 0:
    print("Error: some source paths are too long")
    print("  Long paths can be problematic when building on Windows")
    print("  Please look at the output above and trim the longest paths")
    sys.exit(1)


if __name__ == "__main__":
  main(parse_arguments())
