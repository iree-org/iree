#!/usr/bin/env python3

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Converts a dump of IR text to a markdown file with formatting.

Example usage:
  # Get a dump of IR from a compiler tool:
  $ iree-opt \
    --iree-transformation-pipeline \
    --iree-hal-target-backends=vmvx \
    --mlir-disable-threading \
    --mlir-print-ir-after-all \
    --mlir-print-ir-after-change \
    --mlir-elide-elementsattrs-if-larger=8 \
    $PWD/iree/samples/models/simple_abs.mlir \
    2> /tmp/simple_abs_vmvx_pipeline.mlir
    > /dev/null

  # Convert the IR dump to markdown:
  $ python3 ir_to_markdown.py \
    /tmp/simple_abs_vmvx_pipeline.mlir \
    -o /tmp/simple_abs_vmvx_pipeline.md
"""

import argparse
import re

MLIR_START_SEQUENCE = "// -----//"
MLIR_END_SEQUENCE = "//----- //"


def parse_arguments():
  """Parses command line arguments."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      'input_file_path',
      type=str,
      nargs='?',
      metavar="<input_file_path>",
      help='Input IR dump (.mlir from -mlir-print-ir-after-all)')
  parser.add_argument('-o,',
                      '--output',
                      type=str,
                      required=True,
                      metavar="<output>",
                      help='Output file path (e.g. translation_ir.md)')
  # TODO(scotttodd): flags for original IR path and compilation command line
  #                  .md could then show original IR + flags -> output
  # TODO(scotttodd): flag for markdown flavor (mkdocs, github, etc.)
  # TODO(scotttodd): flag for diff view (correlate IR before and IR after)?

  return parser.parse_args()


def main(args):
  input_file_path = args.input_file_path
  output_file_path = args.output
  print("Converting input file '%s'" % (input_file_path))
  print("     into output file '%s'" % (output_file_path))

  with open(input_file_path, "r") as input_file:
    with open(output_file_path, "w") as output_file:

      # Iterate line by line through the input file, collecting text into
      # blocks and writing them into the output file with markdown formatting
      # as we go.
      #
      # Note: we could parse through and find/replace within the file using
      # regex (or sed), but iterating this way is easier to understand and
      # uses a predictable amount of memory.

      current_block_lines = []
      dump_after_regex = re.compile(MLIR_START_SEQUENCE + "\s(.*)\s" +
                                    MLIR_END_SEQUENCE)

      def finish_block():
        nonlocal current_block_lines
        if len(current_block_lines) != 0:
          current_block_lines.append("```\n\n")
          output_file.writelines(current_block_lines)
          current_block_lines = []

      for input_line in input_file:
        if input_line == "\n":
          continue

        if input_line.startswith(MLIR_START_SEQUENCE):
          finish_block()
          header_text = dump_after_regex.match(input_line).group(1)
          current_block_lines.append("### " + header_text + "\n\n")
          current_block_lines.append("```mlir\n")
        else:
          current_block_lines.append(input_line)

      finish_block()


if __name__ == '__main__':
  main(parse_arguments())
