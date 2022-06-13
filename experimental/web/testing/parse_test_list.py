#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Parses a list of tests from ctest into a format for the web test runner.

Example usage:
  ctest --test-dir build-emscripten --show-only=json-v1 > /tmp/ctest.json
  python3 parse_test_list.py \
      --ctest_dump=/tmp/ctest.json \
      --build_dir=build-emscripten \
      --output_format=html \
      -o /tmp/parsed.html
"""

import argparse
import json
import os


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--ctest_dump",
                      type=str,
                      required=True,
                      help="Path to the output of `ctest --show-only=json-v1`")
  parser.add_argument(
      "--build_dir",
      type=str,
      required=True,
      help="Path to the CMake build directory (absolute or relative)")
  parser.add_argument(
      "--output_format",
      type=str,
      choices=("html", "json"),
      default="html",
      help=
      "Output format, either 'html' for the test runner or 'json' for a list of JSON objects",
  )
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      help="Output file path")
  return parser.parse_args()


def get_normalized_relative_path(absolute_path, root_directory):
  # Strip the root directory prefix and get a relative path.
  relative_path = os.path.relpath(absolute_path, root_directory)
  # Replace the path separator (such as '\' on Windows) with web-style '/'.
  normalized_path = relative_path.replace(os.sep, '/')
  return normalized_path


def parse_ctest_dump(ctest_dump_path, build_dir):
  parsed_tests = []

  # Open the ctest dump JSON file and parse each test.
  # https://cmake.org/cmake/help/latest/manual/ctest.1.html#show-as-json-object-model
  with open(ctest_dump_path, "rt") as f:
    data = json.load(f)
    for test in data["tests"]:
      parsed_test = {
          "testName": test["name"],
          "requiredFiles": [],
          "args": [],
      }

      # Parse the 'command' list into the source file and its arguments.
      #   /path/to/test_runner.js  # such as iree-check-module.js or test.js
      #   arg 1                    # such as --device=local-task
      #   arg 2                    # such as check_vmvx_op.mlir_module.vmfb
      test_source_absolute_path = test["command"][0]
      parsed_test["sourceFile"] = get_normalized_relative_path(
          test_source_absolute_path, build_dir)

      parsed_test["args"] = test["command"][1:]

      # Parse the test "properties".
      # Note: required file paths are relative to the working directory.
      for property in test["properties"]:
        if property["name"] == "REQUIRED_FILES":
          parsed_test["requiredFiles"] = property["value"]
        elif property["name"] == "WORKING_DIRECTORY":
          working_directory_absolute_path = property["value"]
          parsed_test["workingDirectory"] = get_normalized_relative_path(
              working_directory_absolute_path, build_dir)

      parsed_tests.append(parsed_test)

  print("Parsed {} tests from '{}'".format(len(parsed_tests), ctest_dump_path))
  return parsed_tests


def print_parsed_tests(parsed_tests, output_path, output_format):
  with open(output_path, "wt") as f:
    if output_format == "html":
      print("Outputting parsed tests as HTML to '" + output_path + "'")
      for test in parsed_tests:
        f.write(
            "<li><a href=\"test-runner.html?testName={testName}&sourceFile={sourceFile}&workingDirectory={workingDirectory}&requiredFiles={requiredFiles}&args={args}\" target=testRunner>{testName}</a></li>\n"
            .format(testName=test["testName"],
                    sourceFile=test["sourceFile"],
                    workingDirectory=test["workingDirectory"],
                    requiredFiles="[" + ",".join(test["requiredFiles"]) + "]",
                    args="[" + ",".join(test["args"]) + "]"))
    elif output_format == "json":
      print("Outputting parsed tests as JSON to '" + output_path + "'")
      f.write(json.dumps(parsed_tests, indent=2))
    else:
      raise Exception("Unknown output format: '" + output_format + "'")


def main(args):
  # Refine the provided build directory path to a normalized, absolute path.
  build_dir = args.build_dir
  if not os.path.isabs(build_dir):
    build_dir = os.path.join(os.getcwd(), build_dir)
  build_dir = os.path.normpath(build_dir)

  # Create the output directory as needed (relative paths are fine here).
  output_dir = os.path.dirname(args.output)
  if output_dir and not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  parsed_tests = parse_ctest_dump(args.ctest_dump, build_dir)
  parsed_tests.sort(key=lambda test: test["testName"])

  print_parsed_tests(parsed_tests, args.output, args.output_format)


if __name__ == "__main__":
  main(parse_arguments())
