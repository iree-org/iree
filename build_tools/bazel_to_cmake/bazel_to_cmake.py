#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script assists with converting from Bazel BUILD files to CMakeLists.txt.

Bazel BUILD files should, where possible, be written to use simple features
that can be directly evaluated and avoid more advanced features like
variables, list comprehensions, etc.

Generated CMake files will be similar in structure to their source BUILD
files by using the functions in build_tools/cmake/ that imitate corresponding
Bazel rules (e.g. cc_library -> iree_cc_library.cmake).

For usage, see:
  python3 build_tools/bazel_to_cmake/bazel_to_cmake.py --help
"""
# pylint: disable=missing-docstring

import argparse
import datetime
import os
import re
import sys
import textwrap
from enum import Enum

import bazel_to_cmake_converter

repo_root = None

EDIT_BLOCKING_PATTERN = re.compile(
    r"bazel[\s_]*to[\s_]*cmake[\s_]*:?[\s_]*do[\s_]*not[\s_]*edit",
    flags=re.IGNORECASE)


class Status(Enum):
  SUCCEEDED = 1
  FAILED = 2
  SKIPPED = 3
  NO_BUILD_FILE = 4


def parse_arguments():
  global repo_root

  parser = argparse.ArgumentParser(
      description="Bazel to CMake conversion helper.")
  parser.add_argument("--preview",
                      help="Prints results instead of writing files",
                      action="store_true",
                      default=False)
  parser.add_argument(
      "--allow_partial_conversion",
      help="Generates partial files, ignoring errors during conversion.",
      action="store_true",
      default=False)
  parser.add_argument(
      "--verbosity",
      "-v",
      type=int,
      default=0,
      help="Specify verbosity level where higher verbosity emits more logging."
      " 0 (default): Only output errors and summary statistics."
      " 1: Also output the name of each directory as it's being processed and"
      " whether the directory is skipped."
      " 2: Also output when conversion was successful.")

  # Specify only one of these (defaults to --root_dir=iree).
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--dir",
                     help="Converts the BUILD file in the given directory",
                     default=None)
  group.add_argument(
      "--root_dir",
      help="Converts all BUILD files under a root directory (defaults to iree/)",
      default="iree")

  args = parser.parse_args()

  # --dir takes precedence over --root_dir.
  # They are mutually exclusive, but the default value is still set.
  if args.dir:
    args.root_dir = None

  return args


def setup_environment():
  """Sets up some environment globals."""
  global repo_root

  # Determine the repository root (two dir-levels up).
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_relpath(path):
  return os.path.relpath(path, repo_root)


def log(string, *args, indent=0, **kwargs):
  print(textwrap.indent(string, prefix=(indent * " ")),
        *args,
        **kwargs,
        file=sys.stderr)


def convert_directories(directories, write_files, allow_partial_conversion,
                        verbosity):
  failure_dirs = []
  skip_count = 0
  success_count = 0
  for directory in directories:
    status = convert_directory(
        directory,
        write_files=write_files,
        allow_partial_conversion=allow_partial_conversion,
        verbosity=verbosity)
    if status == Status.FAILED:
      failure_dirs.append(repo_relpath(directory))
    elif status == Status.SKIPPED:
      skip_count += 1
    elif status == Status.SUCCEEDED:
      success_count += 1

  log(f"Updated {success_count} and skipped {skip_count} CMakeLists.txt files")
  if failure_dirs:
    log(f"ERROR: Encountered unexpected errors converting {len(failure_dirs)}"
        " directories:")
    log("\n".join(failure_dirs), indent=2)
    sys.exit(1)


def convert_directory(directory_path, write_files, allow_partial_conversion,
                      verbosity):
  if not os.path.isdir(directory_path):
    raise FileNotFoundError(f"Cannot find directory '{directory_path}'")

  rel_dir_path = repo_relpath(directory_path)
  if verbosity >= 1:
    log(f"Processing {rel_dir_path}")

  skip_file_path = os.path.join(directory_path, ".skip_bazel_to_cmake")
  build_file_path = os.path.join(directory_path, "BUILD")
  cmakelists_file_path = os.path.join(directory_path, "CMakeLists.txt")

  rel_cmakelists_file_path = repo_relpath(cmakelists_file_path)
  rel_build_file_path = repo_relpath(build_file_path)

  if os.path.isfile(skip_file_path):
    return Status.SKIPPED
  if not os.path.isfile(build_file_path):
    return Status.NO_BUILD_FILE

  preserve_lines = []
  if os.path.isfile(cmakelists_file_path):
    with open(cmakelists_file_path, "rt") as f:
      found_preserve_marker = False
      for i, line in enumerate(f):
        # Accumulate all lines on and after the special preserve marker.
        if (not found_preserve_marker and
            re.match(r"^### CMAKE PRESERVE ###\s*$", line)):
          found_preserve_marker = True
        if found_preserve_marker:
          preserve_lines.append(line)
          continue

        if EDIT_BLOCKING_PATTERN.search(line):
          if verbosity >= 1:
            log(f"Skipped. line {i + 1}: '{line.strip()}' prevents edits.",
                indent=2)
          return Status.SKIPPED

  header = (f"# Autogenerated from {rel_build_file_path} by\n"
            f"# {repo_relpath(os.path.abspath(__file__))}")

  with open(build_file_path, "rt") as build_file:
    build_file_code = compile(build_file.read(), build_file_path, "exec")
    try:
      converted_text = bazel_to_cmake_converter.convert_build_file(
          build_file_code,
          header,
          allow_partial_conversion=allow_partial_conversion)
      if write_files:
        with open(cmakelists_file_path, "wt") as cmakelists_file:
          cmakelists_file.write(converted_text)
          if preserve_lines:
            cmakelists_file.write("\n")
            cmakelists_file.write("".join(preserve_lines))
      else:
        print(converted_text, end="")
    except (NameError, NotImplementedError) as e:
      log(
          f"ERROR generating {rel_dir_path}.\n"
          f"Missing a rule handler in bazel_to_cmake_converter.py?\n"
          f"Reason: `{type(e).__name__}: {e}`",
          indent=2)
      return Status.FAILED
    except KeyError as e:
      log(
          f"ERROR generating {rel_dir_path}.\n"
          f"Missing a conversion in bazel_to_cmake_targets.py?\n"
          f"Reason: `{type(e).__name__}: {e}`",
          indent=2)
      return Status.FAILED
  if verbosity >= 2:
    log(
        f"Successfly generated {rel_cmakelists_file_path}"
        f" from {rel_build_file_path}",
        indent=2)
  return Status.SUCCEEDED


def main(args):
  """Runs Bazel to CMake conversion."""
  global repo_root

  write_files = not args.preview

  if args.root_dir:
    root_directory_path = os.path.join(repo_root, args.root_dir)
    log(f"Converting directory tree rooted at: {root_directory_path}")
    convert_directories((root for root, _, _ in os.walk(root_directory_path)),
                        write_files=write_files,
                        allow_partial_conversion=args.allow_partial_conversion,
                        verbosity=args.verbosity)
  elif args.dir:
    convert_directories([os.path.join(repo_root, args.dir)],
                        write_files=write_files,
                        allow_partial_conversion=args.allow_partial_conversion,
                        verbosity=args.verbosity)


if __name__ == "__main__":
  setup_environment()
  main(parse_arguments())
