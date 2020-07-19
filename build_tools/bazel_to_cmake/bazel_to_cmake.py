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

import bazel_to_cmake_converter

repo_root = None

EDIT_BLOCKING_PATTERN = re.compile(
    r"bazel[\s_]*to[\s_]*cmake[\s_]*:?[\s_]*do[\s_]*not[\s_]*edit",
    flags=re.IGNORECASE)


def parse_arguments():
  global repo_root

  parser = argparse.ArgumentParser(
      description="Bazel to CMake conversion helper.")
  parser.add_argument(
      "--preview",
      help="Prints results instead of writing files",
      action="store_true",
      default=False)
  parser.add_argument(
      "--allow_partial_conversion",
      help="Generates partial files, ignoring errors during conversion",
      action="store_true",
      default=False)

  # Specify only one of these (defaults to --root_dir=iree).
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--dir",
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


def log(*args, **kwargs):
  print(*args, **kwargs, file=sys.stderr)


def convert_directory_tree(root_directory_path, write_files,
                           allow_partial_conversion):
  log(f"convert_directory_tree: {root_directory_path}")
  for root, _, _ in os.walk(root_directory_path):
    convert_directory(root, write_files, allow_partial_conversion)


def convert_directory(directory_path, write_files, allow_partial_conversion):
  if not os.path.isdir(directory_path):
    raise FileNotFoundError(f"Cannot find directory '{directory_path}'")

  skip_file_path = os.path.join(directory_path, ".skip_bazel_to_cmake")
  build_file_path = os.path.join(directory_path, "BUILD")
  cmakelists_file_path = os.path.join(directory_path, "CMakeLists.txt")

  if os.path.isfile(skip_file_path) or not os.path.isfile(build_file_path):
    # No Bazel BUILD file in this directory or explicit skip.
    return

  global repo_root
  rel_build_file_path = os.path.relpath(build_file_path, repo_root)
  rel_cmakelists_file_path = os.path.relpath(cmakelists_file_path, repo_root)
  log(f"Converting {rel_build_file_path} to {rel_cmakelists_file_path}")

  cmake_file_exists = os.path.isfile(cmakelists_file_path)
  copyright_line = f"# Copyright {datetime.date.today().year} Google LLC"
  write_allowed = write_files
  if cmake_file_exists:
    with open(cmakelists_file_path) as f:
      for i, line in enumerate(f):
        if line.startswith("# Copyright"):
          copyright_line = line.rstrip()
        if EDIT_BLOCKING_PATTERN.search(line):
          log(f"  {rel_cmakelists_file_path} already exists, and "
              f"line {i + 1}: '{line.strip()}' prevents edits. "
              f"Falling back to preview")
          write_allowed = False

  if write_allowed:
    # TODO(scotttodd): Attempt to merge instead of overwrite?
    #   Existing CMakeLists.txt may have special logic that should be preserved
    if cmake_file_exists:
      log(f"  {rel_cmakelists_file_path} already exists; overwriting")
    else:
      log(f"  {rel_cmakelists_file_path} does not exist yet; creating")
  log("")

  with open(build_file_path, "rt") as build_file:
    build_file_code = compile(build_file.read(), build_file_path, "exec")
    try:
      converted_text = bazel_to_cmake_converter.convert_build_file(
          build_file_code,
          copyright_line,
          allow_partial_conversion=allow_partial_conversion)
      if write_allowed:
        with open(cmakelists_file_path, "wt") as cmakelists_file:
          cmakelists_file.write(converted_text)
      else:
        print(converted_text, end="")
    except (NameError, NotImplementedError) as e:
      log(f"Failed to convert {rel_build_file_path}.", end=" ")
      log("Missing a rule handler in bazel_to_cmake.py?")
      log(f"  Reason: `{type(e).__name__}: {e}`")
    except KeyError as e:
      log(f"Failed to convert {rel_build_file_path}.", end=" ")
      log("Missing a conversion in bazel_to_cmake_targets.py?")
      log(f"  Reason: `{type(e).__name__}: {e}`")


def main(args):
  """Runs Bazel to CMake conversion."""
  global repo_root

  write_files = not args.preview

  if args.root_dir:
    convert_directory_tree(
        os.path.join(repo_root, args.root_dir), write_files,
        args.allow_partial_conversion)
  elif args.dir:
    convert_directory(
        os.path.join(repo_root, args.dir), write_files,
        args.allow_partial_conversion)


if __name__ == "__main__":
  setup_environment()
  main(parse_arguments())
