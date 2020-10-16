#!/usr/bin/env python3
#
#===- format_diff.py - Diff Reformatter ----*- python3 -*--===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
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
  svn diff --diff-cmd=diff -x-U0 | python3 format_diff.py -p0 clang-format -i

General usage:
  <some diff> | python3 format_diff.py [--regex] [--lines-style] [-p] binary [args for binary]

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
      "or 'clang-format' then --regex and --lines-style are required.")
  parser.add_argument(
      "--regex",
      metavar="PATTERN",
      default=None,
      help="Regex pattern for selecting file paths to reformat from the piped "
      "diff. This flag is required if 'binary' is not set to 'yapf' or "
      "'clang-format'. Otherwise, this flag overrides the default pattern that "
      "--binary sets.")
  parser.add_argument(
      "--lines-style",
      default=None,
      help="How to style the 'lines' argument for the given binary. Can be set "
      "to 'yapf' or 'clang-format'. This flag is required if 'binary' is not "
      "set to 'yapf' or 'clang-format'.")
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
      raise parser.error("If 'binary' is not 'yapf' or 'clang-format' then "
                         "--regex must be set.")
    if not args.lines_style:
      raise parser.error("If 'binary' is not 'yapf' or 'clang-format' then "
                         "--lines-style must be set.")
  else:
    # Set defaults based off of 'binary'.
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
    match = re.search(fr"^\+\+\+\ (.*?/){{{args.p}}}(\S*)", line)
    if match:
      filename = match.group(2)
    if filename is None:
      continue

    # Match all filenames specified by --regex.
    if not re.match(f"^{args.regex}$", filename):
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
        lines = ["--lines", f"{start_line}-{end_line}"]
      elif args.lines_style == "clang-format":
        lines = ["-lines", f"{start_line}:{end_line}"]
      lines_by_file.setdefault(filename, []).extend(lines)

  # Pass the changed lines to 'binary' alongside any unparsed args (e.g. -i).
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

    # If the formatter printed the formatted code to stdout then print out
    # a unified diff between the formatted and unformatted code.
    # If flags like --verbose are passed to the binary then the diffs this
    # produces won't be particularly helpful.
    formatted_code = io.StringIO(stdout).readlines()
    if len(formatted_code):
      with open(filename) as f:
        unformatted_code = f.readlines()
      diff = difflib.unified_diff(unformatted_code,
                                  formatted_code,
                                  fromfile=filename,
                                  tofile=filename,
                                  fromfiledate="(before formatting)",
                                  tofiledate="(after formatting)")
      diff_string = "".join(diff)
      if len(diff_string) > 0:
        sys.stdout.write(diff_string)


if __name__ == "__main__":
  main()
