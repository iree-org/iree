#!/usr/bin/env python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prepends a license header to files that don't already have one.

By default, only operates on known filetypes but behavior can be overridden with
flags. Ignores files already containing a license as determined by the presence
of a block that looks like "Copyright SOME_YEAR"
"""

import argparse
import datetime
import os
import re
import sys

COPYRIGHT_PATTERN = re.compile(r"Copyright\s+\d+")

LICENSE_HEADER_FORMATTER = """{shebang}{start_comment} Copyright {year} {holder}
{middle_comment} Licensed under the Apache License v2.0 with LLVM Exceptions.
{middle_comment} See https://llvm.org/LICENSE.txt for license information.
{middle_comment} SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception{end_comment}

"""

class CommentSyntax(object):

  def __init__(self, start_comment, middle_comment=None, end_comment=""):
    self.start_comment = start_comment
    self.middle_comment = middle_comment if middle_comment else start_comment
    self.end_comment = end_comment


def comment_arg_parser(v):
  """Can be used to parse a comment syntax triple."""
  if v is None:
    return None
  if not isinstance(v, str):
    raise argparse.ArgumentTypeError("String expected")
  return CommentSyntax(*v.split(","))


def create_multikey(d):
  # pylint: disable=g-complex-comprehension
  return {k: v for keys, v in d.items() for k in keys}


filename_to_comment = create_multikey({
    ("BUILD", "CMakeLists.txt"): CommentSyntax("#"),
})

ext_to_comment = create_multikey({
    (".bzl", ".cfg", ".cmake", ".overlay", ".py", ".sh", ".yml"):
        CommentSyntax("#"),
    (".cc", ".cpp", ".comp", ".fbs", ".h", ".hpp", ".inc", ".td"):
        CommentSyntax("//"),
    (".def",):
        CommentSyntax(";;"),
})


def get_comment_syntax(args):
  """Deterime the comment syntax to use."""
  if args.comment:
    return args.comment
  basename = os.path.basename(args.filename)
  from_filename = filename_to_comment.get(basename)
  if from_filename:
    return from_filename
  _, ext = os.path.splitext(args.filename)
  return ext_to_comment.get(ext, args.default_comment)


def parse_arguments():
  """Parses command line arguments."""
  current_year = datetime.date.today().year
  parser = argparse.ArgumentParser()
  input_group = parser.add_mutually_exclusive_group()
  input_group.add_argument("infile",
                           nargs="?",
                           type=argparse.FileType("r", encoding="UTF-8"),
                           help="Input file to format. Default: stdin",
                           default=sys.stdin)
  parser.add_argument(
      "--filename",
      "--assume-filename",
      type=str,
      default=None,
      help=(
          "Filename to use for determining comment syntax. Default: actual name"
          "of input file."))
  parser.add_argument(
      "--year",
      "-y",
      help="Year to add copyright. Default: the current year ({})".format(
          current_year),
      default=current_year)
  parser.add_argument("--holder",
                      help="Copyright holder. Default: The IREE Authors",
                      default="The IREE Authors")
  parser.add_argument(
      "--quiet",
      help=("Don't raise a runtime error on encountering an unhandled filetype."
            "Useful for running across many files at once. Default: False"),
      action="store_true",
      default=False)
  output_group = parser.add_mutually_exclusive_group()
  output_group.add_argument("-o",
                            "--outfile",
                            "--output",
                            help="File to send output. Default: stdout",
                            type=argparse.FileType("w", encoding="UTF-8"),
                            default=sys.stdout)
  output_group.add_argument("--in_place",
                            "-i",
                            action="store_true",
                            help="Run formatting in place. Default: False",
                            default=False)
  comment_group = parser.add_mutually_exclusive_group()
  comment_group.add_argument("--comment",
                             "-c",
                             type=comment_arg_parser,
                             help="Override comment syntax.",
                             default=None)
  comment_group.add_argument(
      "--default_comment",
      type=comment_arg_parser,
      help="Fallback comment syntax if filename is unknown. Default: None",
      default=None)
  args = parser.parse_args()

  if args.in_place and args.infile == sys.stdin:
    raise parser.error("Cannot format stdin in place")

  if not args.filename and args.infile != sys.stdin:
    args.filename = args.infile.name

  return args


def main(args):
  first_line = args.infile.readline()
  already_has_license = False
  shebang = ""
  content_lines = []
  if first_line.startswith("#!"):
    shebang = first_line
  else:
    content_lines = [first_line]
  content_lines.extend(args.infile.readlines())
  for line in content_lines:
    if COPYRIGHT_PATTERN.search(line):
      already_has_license = True
      break
  if already_has_license:
    header = shebang
  else:
    comment_syntax = get_comment_syntax(args)
    if not comment_syntax:
      if args.quiet:
        header = shebang
      else:
        raise ValueError("Could not determine comment syntax for " +
                         args.filename)
    else:
      header = LICENSE_HEADER_FORMATTER.format(
          # Add a blank line between shebang and license.
          shebang=(shebang + "\n" if shebang else ""),
          start_comment=comment_syntax.start_comment,
          middle_comment=comment_syntax.middle_comment,
          # Add a blank line before the end comment.
          end_comment=("\n" + comment_syntax.end_comment
                       if comment_syntax.end_comment else ""),
          year=args.year,
          holder=args.holder)

  # Have to open for write after we're done reading.
  if args.in_place:
    args.outfile = open(args.filename, "w", encoding="UTF-8")
  args.outfile.write(header)
  args.outfile.writelines(content_lines)


if __name__ == "__main__":
  main(parse_arguments())
