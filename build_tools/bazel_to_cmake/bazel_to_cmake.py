#!/usr/bin/env python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This script assists with converting from Bazel BUILD files to CMakeLists.txt.

Bazel BUILD files should, where possible, be written to use simple features
that can be directly evaluated and avoid more advanced features like
variables, list comprehensions, etc.

Generated CMake files will be similar in structure to their source BUILD
files by using the functions in build_tools/cmake/ that imitate corresponding
Bazel rules (e.g. cc_library -> iree_cc_library.cmake).

For usage, see:
  python3 build_tools/bazel_to_cmake/bazel_to_cmake.py --help

Configuration
-------------
When invoked, bazel_to_cmake will traverse up from the current directory until
it finds a ".bazel_to_cmake.cfg.py" file. This file both serves as a marker
for the repository root and provides repository specific configuration.

The file is evaluated as a module and can have the following customizations:

* DEFAULT_ROOT_DIRS: A list of root directory names that should be processed
  (relative to the repository root) when invoked without a --repo_root or --dir.
* REPO_MAP: Mapping of canonical Bazel repo name (i.e. "@iree_core") to what it
  is known as locally (most commonly the empty string). This is used in global
  target rules to make sure that they work either in the defining or referencing
  repository.
* CustomBuildFileFunctions: A class that extends
  `bazel_to_cmake_converter.BuildFileFunctions` and injects globals for
  processing the BUILD file. All symbols that do not start with "_" are
  available.
* CustomTargetConverter: A class that extends
  `bazel_to_cmake_targets.TargetConverter` and customizes target mapping.
  Typically, this is used for purely local targets in leaf projects (as global
  targets will be encoded in the main bazel_to_cmake_targets.py file).
"""
# pylint: disable=missing-docstring

import argparse
import datetime
import importlib
import importlib.util
import os
import re
import sys
import textwrap
import types
from enum import Enum

import bazel_to_cmake_converter

repo_root = None
repo_cfg = None

EDIT_BLOCKING_PATTERN = re.compile(
    r"bazel[\s_]*to[\s_]*cmake[\s_]*:?[\s_]*do[\s_]*not[\s_]*edit",
    flags=re.IGNORECASE)

PRESERVE_TAG = "### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###"
REPO_CFG_FILE = ".bazel_to_cmake.cfg.py"
REPO_CFG_MODULE_NAME = "bazel_to_cmake_repo_config"


class Status(Enum):
  UPDATED = 1
  NOOP = 2
  FAILED = 3
  SKIPPED = 4
  NO_BUILD_FILE = 5


def parse_arguments():
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

  # Specify only one of these (defaults to --root_dir=<main source dirs>).
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--dir",
                     help="Converts the BUILD file in the given directory",
                     default=None)
  default_root_dirs = (repo_cfg.DEFAULT_ROOT_DIRS if hasattr(
      repo_cfg, "DEFAULT_ROOT_DIRS") else [])
  group.add_argument("--root_dir",
                     nargs="+",
                     help="Converts all BUILD files under a root directory",
                     default=default_root_dirs)

  args = parser.parse_args()

  # --dir takes precedence over --root_dir.
  # They are mutually exclusive, but the default value is still set.
  if args.dir:
    args.root_dir = None

  return args


def setup_environment():
  """Sets up some environment globals."""
  global repo_root
  global repo_cfg

  # Scan up the directory tree for a repo config file.
  check_dir = os.getcwd()
  while not os.path.exists(os.path.join(check_dir, REPO_CFG_FILE)):
    new_check_dir = os.path.dirname(check_dir)
    if not new_check_dir or new_check_dir == check_dir:
      print(f"ERROR: Could not find {REPO_CFG_FILE} in a parent directory "
            f"of {os.getcwd()}")
      sys.exit(1)
    check_dir = new_check_dir
  repo_root = check_dir
  log(f"Using repo root {repo_root}")

  # Dynamically load the config file as a module.
  orig_dont_write_bytecode = sys.dont_write_bytecode
  sys.dont_write_bytecode = True  # Don't generate __pycache__ dir
  spec = importlib.util.spec_from_file_location(
      REPO_CFG_MODULE_NAME, os.path.join(repo_root, REPO_CFG_FILE))
  repo_cfg = importlib.util.module_from_spec(spec)
  sys.modules[REPO_CFG_MODULE_NAME] = repo_cfg
  spec.loader.exec_module(repo_cfg)
  sys.dont_write_bytecode = orig_dont_write_bytecode


def repo_relpath(path):
  return os.path.relpath(path, repo_root).replace("\\", "/")


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
  noop_count = 0
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
    elif status == Status.UPDATED:
      success_count += 1
    elif status == Status.NOOP:
      noop_count += 1

  log(f"{success_count} CMakeLists.txt files were updated, {skip_count} were"
      f" skipped, and {noop_count} required no change.")
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

  # Scan for a BUILD file.
  build_file_found = False
  build_file_basenames = ["BUILD", "BUILD.bazel"]
  for build_file_basename in build_file_basenames:
    build_file_path = os.path.join(directory_path, build_file_basename)

    rel_build_file_path = repo_relpath(build_file_path)
    if os.path.isfile(build_file_path):
      build_file_found = True
      break
  cmakelists_file_path = os.path.join(directory_path, "CMakeLists.txt")
  rel_cmakelists_file_path = repo_relpath(cmakelists_file_path)

  if not build_file_found:
    return Status.NO_BUILD_FILE

  autogeneration_tag = f"Autogenerated by {repo_relpath(os.path.abspath(__file__))}"

  header = "\n".join(["#" * 80] + [
      l.ljust(79) + "#" for l in [
          f"# {autogeneration_tag} from",
          f"# {rel_build_file_path}",
          "#",
          "# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary",
          "# CMake-only content.",
          "#",
          f"# To disable autogeneration for this file entirely, delete this header.",
      ]
  ] + ["#" * 80])

  old_lines = []
  preserved_footer_lines = ["\n" + PRESERVE_TAG + "\n"]

  # Read CMakeLists.txt and check if it has the auto-generated header.
  if os.path.isfile(cmakelists_file_path):
    found_autogeneration_tag = False
    found_preserve_tag = False
    with open(cmakelists_file_path) as f:
      old_lines = f.readlines()

    for line in old_lines:
      if not found_autogeneration_tag and autogeneration_tag in line:
        found_autogeneration_tag = True
      if not found_preserve_tag and PRESERVE_TAG in line:
        found_preserve_tag = True
      elif found_preserve_tag:
        preserved_footer_lines.append(line)
    if not found_autogeneration_tag:
      if verbosity >= 1:
        log(f"Skipped. Did not find autogeneration line.", indent=2)
      return Status.SKIPPED
  preserved_footer = "".join(preserved_footer_lines)

  # Read the Bazel BUILD file and interpret it.
  with open(build_file_path, "rt") as build_file:
    build_file_contents = build_file.read()
  if "bazel-to-cmake: skip" in build_file_contents:
    return Status.SKIPPED
  build_file_code = compile(build_file_contents, build_file_path, "exec")
  try:
    converted_build_file = bazel_to_cmake_converter.convert_build_file(
        build_file_code,
        repo_cfg=repo_cfg,
        allow_partial_conversion=allow_partial_conversion)
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
  converted_content = header + converted_build_file + preserved_footer
  if write_files:
    with open(cmakelists_file_path, "wt") as cmakelists_file:
      cmakelists_file.write(converted_content)
  else:
    print(converted_content, end="")

  if converted_content == "".join(old_lines):
    if verbosity >= 2:
      log(f"{rel_cmakelists_file_path} required no update", indent=2)
    return Status.NOOP

  if verbosity >= 2:
    log(
        f"Successfly generated {rel_cmakelists_file_path}"
        f" from {rel_build_file_path}",
        indent=2)
  return Status.UPDATED


def main(args):
  """Runs Bazel to CMake conversion."""
  global repo_root

  write_files = not args.preview

  if args.root_dir:
    for root_dir in args.root_dir:
      root_directory_path = os.path.join(repo_root, root_dir)
      log(f"Converting directory tree rooted at: {root_directory_path}")
      convert_directories(
          (root for root, _, _ in os.walk(root_directory_path)),
          write_files=write_files,
          allow_partial_conversion=args.allow_partial_conversion,
          verbosity=args.verbosity)
  elif args.dir:
    convert_directories([os.path.join(repo_root, args.dir)],
                        write_files=write_files,
                        allow_partial_conversion=args.allow_partial_conversion,
                        verbosity=args.verbosity)
  else:
    log(f"ERROR: None of --root-dir, --dir arguments or DEFAULT_ROOT_DIRS in "
        f".bazel_to_cmake.cfg.py: No conversion will be done")
    sys.exit(1)


if __name__ == "__main__":
  setup_environment()
  main(parse_arguments())
