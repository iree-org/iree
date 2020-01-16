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

# This script assists with converting from Bazel BUILD files to CMakeLists.txt.
#
# Bazel BUILD files should, where possible, be written to use simple features
# that can be directly evaluated and avoid more advanced features like
# variables, list comprehensions, etc.
#
# Generated CMake files will be similar in structure to their source BUILD
# files by using the functions in build_tools/cmake/ that imitate corresponding
# Bazel rules (i.e. cc_library -> iree_cc_library.cmake).
#
# Usage:
#   python3 build_tools/scripts/bazel_to_cmake.py

import os
import textwrap

BUILD_FILE_NAME = "BUILD"
CMAKELISTS_FILE_NAME = "CMakeLists.txt"


class BuildFileFunctions(object):

  def __init__(self, converter):
    self.converter = converter

  # ------------------------------------------------------------------------- #
  # Conversion utilities, written to reduce boilerplate and allow for reuse   #
  # between similar rule conversions (e.g. cc_library and cc_binary).         #
  # ------------------------------------------------------------------------- #

  def _convert_name_block(self, **kwargs):
    #  NAME
    #    rule_name
    return "  NAME\n    %s\n" % (kwargs["name"])

  def _convert_testonly_block(self, **kwargs):
    if kwargs.get("testonly"):
      # Note: this is a truthiness check as well as an existence check, i.e.
      # Bazel `testonly = False` will be handled correctly by this condition.
      return "  TESTONLY\n"
    else:
      return ""

  def _convert_filelist_block(self, list_name, files):
    if not files:
      return ""

    #  list_name
    #    "file_1.h"
    #    "file_2.h"
    #    "file_3.h"
    files_list = "\n".join(["    \"%s\"" % (file) for file in files])
    return "  %s\n%s\n" % (list_name, files_list)

  def _convert_hdrs_block(self, **kwargs):
    return self._convert_filelist_block("HDRS", kwargs.get("hdrs"))

  def _convert_srcs_block(self, **kwargs):
    return self._convert_filelist_block("SRCS", kwargs.get("srcs"))

  def _convert_dep(self, dep):
    # TODO(scotttodd): special cases where pattern isn't found
    #   could import from a separate file to keep this file more generic

    # Bazel `//iree/base`     -> CMake `iree::base`
    # Bazel `//iree/base:api` -> CMake `iree::base::api`
    dep = dep.replace("//", "")  # iree/base:api
    dep = dep.replace(":", "::")  # iree/base::api
    dep = dep.replace("/", "::")  # iree::base::api
    return dep

  def _convert_deps_block(self, **kwargs):
    if not kwargs.get("deps"):
      return ""

    #  DEPS
    #    dep1::target1
    #    dep2::target2
    deps = kwargs.get("deps")
    deps_list = "\n".join(["    %s" % (self._convert_dep(dep)) for dep in deps])
    return "  DEPS\n%s\n" % (deps_list)

  # ------------------------------------------------------------------------- #
  # Function handlers that convert BUILD definitions to CMake definitions.    #
  #                                                                           #
  # Names and signatures must match 1:1 with those expected in BUILD files.   #
  # Each function that may be found in a BUILD file must be listed here.      #
  # ------------------------------------------------------------------------- #

  def load(self, *args):
    pass

  def package(self, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def cc_library(self, **kwargs):
    name_block = self._convert_name_block(**kwargs)
    hdrs_block = self._convert_hdrs_block(**kwargs)
    srcs_block = self._convert_srcs_block(**kwargs)
    deps_block = self._convert_deps_block(**kwargs)
    testonly_block = self._convert_testonly_block(**kwargs)

    self.converter.body += """iree_cc_library(
%(name_block)s%(hdrs_block)s%(srcs_block)s%(deps_block)s%(testonly_block)s  PUBLIC
)\n\n""" % {
    "name_block": name_block,
    "hdrs_block": hdrs_block,
    "srcs_block": srcs_block,
    "deps_block": deps_block,
    "testonly_block": testonly_block,
    }


class Converter(object):

  def __init__(self):
    self.body = ""

  def convert(self):
    return self.template % {"body": self.body}

  # TODO(scotttodd): Write license header with the current year
  template = textwrap.dedent("""\
    # This file was generated using build_tools/scripts/bazel_to_cmake.py.

    %(body)s""")


def GetDict(obj):
  ret = {}
  for k in dir(obj):
    if not k.startswith("_"):
      ret[k] = getattr(obj, k)
  return ret


def convert_directory_tree(root_directory_path):
  print("convert_directory_tree: %s" % (root_directory_path))
  for root, directory_names, file_names in os.walk(root_directory_path):
    convert_directory(root)


def convert_directory(directory_path):
  print("convert_directory: %s" % (directory_path))

  build_file_path = os.path.join(directory_path, BUILD_FILE_NAME)
  cmakelists_file_path = os.path.join(directory_path, CMAKELISTS_FILE_NAME)

  if not os.path.isfile(build_file_path):
    # No Bazel BUILD file in this directory to convert, skip.
    return

  # TODO(scotttodd): Attempt to merge instead of overwrite?
  #   Existing CMakeLists.txt may have special logic that should be preserved
  if os.path.isfile(cmakelists_file_path):
    print("  Already has CMakeLists.txt")
  else:
    print("  Does not have CMakeLists.txt")

  with open(build_file_path) as build_file:
    build_file_code = compile(build_file.read(), build_file_path, "exec")
    converter = Converter()
    exec(build_file_code, GetDict(BuildFileFunctions(converter)))
    converted_text = converter.convert()
    print(converted_text)
    # TODO(scotttodd): write to file


def run():
  """Runs Bazel to CMake conversion."""

  # Determine the repository root (two dir-levels up).
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  print("Repository root: %s" % (repo_root))

  # TODO(scotttodd): Flags:
  #   debug: print to console, do not write files
  #   check: return success code if files match, failure with diff otherwise
  #   path: individual build file or directory to run on

  # convert_directory_tree(os.path.join(repo_root, "iree"))
  convert_directory(os.path.join(repo_root, "iree/hal/testing"))


if __name__ == "__main__":
  run()
