#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import io
import os
import re
import sys
from itertools import chain

import xnncommon


def key_value_pair(line):
  key, value = line.split("=", 1)
  # represent value as integer, if possible, otherwise as str
  try:
    value = int(value)
  except ValueError:
    pass
  return key, value


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("input", metavar="FILE", nargs=1,
          help="Input file")
parser.add_argument("-D", dest="defines", metavar="KEY=VALUE", nargs="*",
          type=key_value_pair, action="append",
          help="Predefined variables")
parser.add_argument("-o", "--output",
          help='Output file')
parser.set_defaults(defines=list())


LEADING_WHITESPACE_REGEX = re.compile(r"^\s*", flags=0)


def extract_leading_whitespace(line):
  match = re.match(r"\s*", line)
  return match.group(0) if match else ""


def escape(line):
  output_parts = []
  while "${" in line:
    start_pos = line.index("${")
    end_pos = line.index("}", start_pos + 2)
    if start_pos != 0:
      output_parts.append("\"" + line[:start_pos].replace("\"", "\\\"") + "\"")
    output_parts.append("str(" + line[start_pos+2:end_pos] + ")")
    line = line[end_pos+1:]
  if line:
    output_parts.append("\"" + line.replace("\"", "\\\"") + "\"")
  return " + ".join(output_parts)


def preprocess(input_text, input_globals, input_path="codegen"):
  input_lines = input_text.splitlines()
  python_lines = []

  blank_lines = 0

  last_line = ""
  last_indent = ""

  # List of tuples (total_index, python_indent)
  indent_stack = [("", "")]

  # Indicates whether this is the first line inside Python
  # code block (i.e. for, while, if, elif, else)
  python_block_start = True
  for i, input_line in enumerate(input_lines):
    if input_line == "":
      blank_lines += 1
      continue
    # Skip lint markers.
    if 'LINT' in input_line:
      continue

    input_indent = extract_leading_whitespace(input_line)
    if python_block_start:
      assert input_indent.startswith(last_indent)
      extra_python_indent = input_indent[len(last_indent):]
      python_indent = indent_stack[-1][1] + extra_python_indent
      indent_stack.append((input_indent, python_indent))
      assert input_indent.startswith(indent_stack[-1][0])
    else:
      while not input_indent.startswith(indent_stack[-1][0]):
        del indent_stack[-1]
    python_block_start = False

    python_indent = indent_stack[-1][1]
    stripped_input_line = input_line.strip()
    if stripped_input_line.startswith("$") and not stripped_input_line.startswith("${"):
      if stripped_input_line.endswith(":"):
        python_block_start = True
      while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1
      python_lines.append(python_indent + stripped_input_line.replace("$", ""))
    else:
      assert input_line.startswith(python_indent)
      while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1
      python_lines.append(python_indent + "print(%s, file=OUT_STREAM)" % escape(input_line[len(python_indent):]))
    last_line = input_line
    last_indent = input_indent

  while blank_lines != 0:
    python_lines.append(python_indent + "print(file=OUT_STREAM)")
    blank_lines -= 1

  exec_globals = dict(input_globals)
  if sys.version_info > (3, 0):
    output_stream = io.StringIO()
  else:
    output_stream = io.BytesIO()
  exec_globals["OUT_STREAM"] = output_stream
  python_bytecode = compile("\n".join(python_lines), input_path, 'exec')
  exec(python_bytecode, exec_globals)

  return output_stream.getvalue()


PREAMBLE = """\
// Auto-generated file. Do not edit!
//   Template: {template}
//   Generator: {generator}
//
"""


def main(args):
  options = parser.parse_args(args)

  input_text = codecs.open(options.input[0], "r", encoding="utf-8").read()
  python_globals = dict(chain(*options.defines))
  output_text = PREAMBLE.format(template=options.input[0], generator=sys.argv[0]) + preprocess(input_text, python_globals, options.input[0])

  xnncommon.overwrite_if_changed(options.output, output_text)

if __name__ == "__main__":
  main(sys.argv[1:])
