#!/usr/bin/env python
#
#===- yapf_format_diff.py - YAPF Diff Reformatter ----*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
"""
This script reads input from a unified diff and reformats all the changed
lines. This is useful to reformat all the lines touched by a specific patch.
Example usage:

  git diff -U0 HEAD^ | python3 format_diff.py yapf -i
  git diff -U0 HEAD^ | python3 format_diff.py clang-format -i

  svn diff --diff-cmd=diff -x-U0 | format_diff.py clang-format -i -p0

It should be noted that the filename contained in the diff is used unmodified
to determine the source file to update. Users calling this script directly
should be careful to ensure that the path in the diff is correct relative to the
current working directory.
"""

import argparse
import difflib
import io
import re
import subprocess
import sys

BINARY_TO_DEFAULT_REGEX = {
    "yapf": r".*\.py",
    "clang-format":
        r".*\.(cpp|cc|c\+\+|cxx|c|cl|h|hh|hpp|hxx|m|mm|inc|js|ts|proto|"
        r"protodevel|java|cs)",
}


def parse_arguments():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "binary",
      help="Location of binary to use for formatting. This controls the "
      "default values of --regex and --lines-style. If binary isn't 'yapf' "
      "or 'clang-format' then --regex and --lines-style must be specified.")
  parser.add_argument(
      "--regex",
      metavar="PATTERN",
      default=None,
      help="Custom pattern selecting file paths from the diff to reformat. "
      "Must be specified if --binary is not 'yapf' or 'clang-format', and "
      "overrides the default --binary sets otherwise. (case sensitive)")
  parser.add_argument(
      "--lines-style",
      default=None,
      help="How to style the --lines argument for --binary. Must be one of "
      "Either 'yapf' or 'clang-format'. This must be set manually if --binary "
      "is not 'yapf' or 'clang-format'.")
  parser.add_argument(
      "-p",
      metavar="NUM",
      default=1,
      help="Strip the smallest prefix containing P slashes. Set to 0 if "
      "passing `--no-prefix` to `git diff` or using `svn`")

  # Parse and error-check arguments
  args, binary_args = parser.parse_known_args()
  if args.binary not in BINARY_TO_DEFAULT_REGEX:
    if not args.regex:
      raise parser.error("If --binary is not 'yapf' or 'clang-format' then "
                         "--regex must be set.")
    if not args.lines_style:
      raise parser.error("If --binary is not 'yapf' or 'clang-format' then "
                         "--lines-style must be set.")
  else:
    # Set defaults based off of --binary.
    if not args.regex:
      args.regex = BINARY_TO_DEFAULT_REGEX[args.binary]
    if not args.lines_style:
      args.lines_style = args.binary

  if args.lines_style not in ["yapf", "clang-format"]:
    raise parser.error(f"Unexpected value for --line-style {args.lines_style}")

  return args, binary_args


def main():
  args, binary_args = parse_arguments()

  # Extract changed lines for each file.
  filename = None
  lines_by_file = {}
  for line in sys.stdin:
    # Match all filenames.
    match = re.search(r"^\+\+\+\ (.*?/){%s}(\S*)" % args.p, line)
    if match:
      filename = match.group(2)
    if filename is None:
      continue

    # Match all filenames specified by --regex.
    if not re.match("^%s$" % args.regex, filename):
      continue

    # Match unified diff line numbers.
    match = re.search(r"^@@.*\+(\d+)(,(\d+))?", line)
    if match:
      start_line = int(match.group(1))
      line_count = 1
      if match.group(3):
        line_count = int(match.group(3))
      if line_count == 0:
        continue
      end_line = start_line + line_count - 1

      if args.lines_style == "yapf":
        lines = ["--lines", str(start_line) + "-" + str(end_line)]
      elif args.lines_style == "clang-format":
        lines = ['-lines', str(start_line) + ':' + str(end_line)]
      lines_by_file.setdefault(filename, []).extend(lines)

  # Pass the changed lines to --binary alongside any 'unknown' args (e.g. -i).
  for filename, lines in lines_by_file.items():
    command = [args.binary, filename]
    command.extend(lines)
    command.extend(binary_args)

    print(f"Running `{' '.join(command)}`")
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=None,
                         stdin=subprocess.PIPE,
                         universal_newlines=True)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
      sys.exit(p.returncode)

    # Print --binary's output if in-place formatting isn't specified.
    if "-i" not in binary_args:
      with open(filename) as f:
        code = f.readlines()
      formatted_code = io.StringIO(stdout).readlines()
      diff = difflib.unified_diff(code, formatted_code, filename, filename,
                                  "(before formatting)", "(after formatting)")
      diff_string = "".join(diff)
      if len(diff_string) > 0:
        sys.stdout.write(diff_string)


if __name__ == "__main__":
  main()
