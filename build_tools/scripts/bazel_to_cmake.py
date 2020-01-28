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
# Bazel rules (e.g. cc_library -> iree_cc_library.cmake).
#
# For usage, see:
#   python3 build_tools/scripts/bazel_to_cmake.py --help

import argparse
import bazel_to_cmake_targets
import datetime
import os
import textwrap
from collections import OrderedDict
from itertools import repeat
import glob
import re

repo_root = None

EDIT_BLOCKING_PATTERN = re.compile(
    "bazel[\s_]*to[\s_]*cmake[\s_]*:?[\s_]*do[\s_]*not[\s_]*edit",
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

  # TODO(scotttodd): --check option that returns success/failure depending on
  #   if files match the converted versions

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


class BuildFileFunctions(object):
  """Object passed to `exec` that has handlers for BUILD file functions."""

  def __init__(self, converter):
    self.converter = converter

  # ------------------------------------------------------------------------- #
  # Conversion utilities, written to reduce boilerplate and allow for reuse   #
  # between similar rule conversions (e.g. cc_library and cc_binary).         #
  # ------------------------------------------------------------------------- #

  def _convert_name_block(self, name):
    #  NAME
    #    rule_name
    return "  NAME\n    %s\n" % (name)

  def _convert_cc_namespace_block(self, cc_namespace):
    #  CC_NAMESPACE
    #    "cc_namespace"
    if not cc_namespace:
      return ""
    return "  CC_NAMESPACE\n    \"%s\"\n" % (cc_namespace)

  def _convert_cpp_namespace_block(self, cpp_namespace):
    if not cpp_namespace:
      return ""
    #  CPP_NAMESPACE
    #    "cpp_namespace"
    return "  CPP_NAMESPACE\n    \"%s\"\n" % (cpp_namespace)

  def _convert_translation_block(self, translation):
    return "  TRANSLATION\n    \"%s\"\n" % (translation)

  def _convert_translate_tool_block(self, translate_tool):
    if translate_tool:
      # Bazel `//iree/base`     -> CMake `iree::base`
      # Bazel `//iree/base:api` -> CMake `iree::base::api`
      translate_tool = translate_tool.replace("//", "")  # iree/base:api
      translate_tool = translate_tool.replace(":", "_")  # iree/base::api
      translate_tool = translate_tool.replace("/", "_")  # iree::base::api
      return "  TRANSLATE_TOOL\n    %s\n" % (translate_tool)
    else:
      return ""

  def _convert_option_block(self, option, option_value):
    if option_value:
      # Note: this is a truthiness check as well as an existence check, i.e.
      # Bazel `testonly = False` will be handled correctly by this condition.
      return "  %s\n" % option
    else:
      return ""

  def _convert_alwayslink_block(self, alwayslink):
    return self._convert_option_block("ALWAYSLINK", alwayslink)

  def _convert_testonly_block(self, testonly):
    return self._convert_option_block("TESTONLY", testonly)

  def _convert_flatten_block(self, flatten):
    return self._convert_option_block("FLATTEN", flatten)

  def _convert_filelist_block(self, list_name, files):
    if not files:
      return ""

    #  list_name
    #    "file_1.h"
    #    "file_2.h"
    #    "file_3.h"
    files_list = "\n".join(["    \"%s\"" % (file) for file in files])
    return "  %s\n%s\n" % (list_name, files_list)

  def _convert_hdrs_block(self, hdrs):
    return self._convert_filelist_block("HDRS", hdrs)

  def _convert_srcs_block(self, srcs):
    return self._convert_filelist_block("SRCS", srcs)

  def _convert_src_block(self, src):
    return "  SRC\n    \"%s\"\n" % src

  def _convert_cc_file_output_block(self, cc_file_output):
    return "  CC_FILE_OUTPUT\n    \"%s\"\n" % (cc_file_output)

  def _convert_h_file_output_block(self, h_file_output):
    return "  H_FILE_OUTPUT\n    \"%s\"\n" % (h_file_output)

  def _convert_td_file_block(self, td_file):
    if td_file.startswith("//iree"):
      # Bazel `//iree/dir/td_file.td`
      # -> CMake `${IREE_ROOT_DIR}/iree/dir/td_file.td
      # Bazel `//iree/dir/IR:td_file.td`
      # -> CMake `${IREE_ROOT_DIR}/iree/dir/IR/td_file.td
      td_file = td_file.replace("//", "${IREE_ROOT_DIR}/")
      td_file = td_file.replace(":", "/")
    return "  TD_FILE\n    \"%s\"\n" % (td_file)

  def _convert_tbl_outs_block(self, tbl_outs):
    outs_list = "\n".join(["    %s %s" % tbl_out for tbl_out in tbl_outs])
    return "  OUTS\n%s\n" % (outs_list)

  def _convert_tblgen_block(self, tblgen):
    if tblgen.endswith("iree-tblgen"):
      return "  TBLGEN\n    IREE\n"
    else:
      return ""

  def _convert_target(self, target):
    if target.startswith(":"):
      # Bazel package-relative `:logging` -> CMake absolute `iree::base::logging`
      package = os.path.dirname(self.converter.rel_build_file_path)
      package = package.replace(os.path.sep, "::")
      if package.endswith(target):
        target = package  # Omit target if it matches the package name
      else:
        target = package + ":" + target
      if target.endswith("_gen"):
        # Files created by gentbl have to be included as source and header files
        # and not as a dependency. Adding these targets to the dependencies list,
        # results in linkage failures if the library including the gentbl dep is
        # marked as ALWAYSLINK.
        # TODO: Some targets not to be included end to "Gen", but others like
        #       LLVMTableGen still need to be included.
        return ""
    elif not target.startswith("//iree"):
      # External target, call helper method for special case handling.
      target = bazel_to_cmake_targets.convert_external_target(target)
    else:
      # Bazel `//iree/base`     -> CMake `iree::base`
      # Bazel `//iree/base:api` -> CMake `iree::base::api`
      target = target.replace("//", "")  # iree/base:api
      target = target.replace(":", "::")  # iree/base::api
      target = target.replace("/", "::")  # iree::base::api
    return target

  def _convert_data_block(self, data):
    if not data:
      return ""

    #  DATA
    #    package1::target1
    #    package1::target2
    #    package2::target
    data_list = [self._convert_target(dep) for dep in data]
    # Remove Falsey (None and empty string) values and duplicates, preserving the original ordering.
    data_list = list(filter(None, OrderedDict(zip(data_list, repeat(None)))))
    data_list = "\n".join(["    %s" % (data,) for data in data_list])
    return "  DATA\n%s\n" % (data_list,)

  def _convert_deps_block(self, deps):
    if not deps:
      return ""

    #  DEPS
    #    package1::target1
    #    package1::target2
    #    package2::target
    deps_list = [self._convert_target(dep) for dep in deps]
    # Remove Falsey (None and empty string) values and duplicates, preserving the original ordering.
    deps_list = list(filter(None, OrderedDict(zip(deps_list, repeat(None)))))
    deps_list = "\n".join(["    %s" % (dep,) for dep in deps_list])
    return "  DEPS\n%s\n" % (deps_list,)

  def _convert_unimplemented_function(self, rule, *args, **kwargs):
    name = kwargs.get("name", "unnamed")
    self.converter.body += "# Unimplemented %(rule)s %(name)s\n" % {
        "rule": rule,
        "name": name
    }

  # ------------------------------------------------------------------------- #
  # Function handlers that convert BUILD definitions to CMake definitions.    #
  #                                                                           #
  # Names and signatures must match 1:1 with those expected in BUILD files.   #
  # Each function that may be found in a BUILD file must be listed here.      #
  # ------------------------------------------------------------------------- #

  def load(self, *args):
    # No mapping to CMake, ignore.
    pass

  def package(self, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def iree_build_test(self, **kwargs):
    pass

  def filegroup(self, **kwargs):
    # Not implemented yet. Might be a no-op, or may want to evaluate the srcs
    # attribute and pass them along to any targets that depend on the filegroup.
    # Cross-package dependencies and complicated globs could be hard to handle.
    self._convert_unimplemented_function("filegroup", **kwargs)

  def exports_files(self, *args, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def glob(self, include, exclude=[], exclude_directories=1):
    # Rather than converting bazel globs into CMake globs, we evaluate the glob at
    # conversion time. This avoids issues with different glob semantics and dire
    # warnings about not knowing when to reevaluate the glob.
    # See https://cmake.org/cmake/help/v3.12/command/file.html#filesystem
    if exclude_directories != 1:
      raise ValueError("Non-default exclude_directories not supported")

    filepaths = []
    for pattern in include:
      if "**" in pattern:
        # bazel's glob has some specific restrictions about crossing package boundaries.
        # We have no uses of recursive globs. Rather than try to emulate them or
        # silently give different behavior, just error out.
        # See https://docs.bazel.build/versions/master/be/functions.html#glob
        raise ValueError("Recursive globs not supported")

      filepaths += glob.glob(self.converter.directory_path + "/" + pattern)

    exclude_filepaths = set([])
    for pattern in exclude:
      if "**" in pattern:
        # See comment above
        raise ValueError("Recursive globs not supported")
      exclude_filepaths.update(
          glob.glob(self.converter.directory_path + "/" + pattern))

    basenames = sorted([
        os.path.basename(path)
        for path in filepaths
        if path not in exclude_filepaths
    ])
    return basenames

  def config_setting(self, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def cc_library(self,
                 name,
                 hdrs=[],
                 srcs=[],
                 deps=[],
                 alwayslink=False,
                 testonly=False,
                 **kwargs):
    name_block = self._convert_name_block(name)
    hdrs_block = self._convert_hdrs_block(hdrs)
    srcs_block = self._convert_srcs_block(srcs)
    deps_block = self._convert_deps_block(deps)
    alwayslink_block = self._convert_alwayslink_block(alwayslink)
    testonly_block = self._convert_testonly_block(testonly)

    self.converter.body += """iree_cc_library(
%(name_block)s%(hdrs_block)s%(srcs_block)s%(deps_block)s%(alwayslink_block)s%(testonly_block)s  PUBLIC
)\n\n""" % {
    "name_block": name_block,
    "hdrs_block": hdrs_block,
    "srcs_block": srcs_block,
    "deps_block": deps_block,
    "alwayslink_block": alwayslink_block,
    "testonly_block": testonly_block,
    }

  def cc_test(self, name, hdrs=[], srcs=[], deps=[], **kwargs):
    name_block = self._convert_name_block(name)
    hdrs_block = self._convert_hdrs_block(hdrs)
    srcs_block = self._convert_srcs_block(srcs)
    deps_block = self._convert_deps_block(deps)

    self.converter.body += """iree_cc_test(
%(name_block)s%(hdrs_block)s%(srcs_block)s%(deps_block)s)\n\n""" % {
    "name_block": name_block,
    "hdrs_block": hdrs_block,
    "srcs_block": srcs_block,
    "deps_block": deps_block,
    }

  def cc_binary(self, name, srcs=[], deps=[], **kwargs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    deps_block = self._convert_deps_block(deps)

    self.converter.body += """iree_cc_binary(
%(name_block)s%(srcs_block)s%(deps_block)s)\n\n""" % {
    "name_block": name_block,
    "srcs_block": srcs_block,
    "deps_block": deps_block,
    }

  def cc_embed_data(self,
                    name,
                    srcs,
                    cc_file_output,
                    h_file_output,
                    cpp_namespace=None,
                    strip_prefix=None,
                    flatten=False,
                    identifier=None,
                    **kwargs):
    if identifier:
      self._convert_unimplemented_function("cc_embed_data", name=name)
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    cc_file_output_block = self._convert_cc_file_output_block(cc_file_output)
    h_file_output_block = self._convert_h_file_output_block(h_file_output)
    namespace_block = self._convert_cpp_namespace_block(cpp_namespace)
    flatten_block = self._convert_flatten_block(flatten)

    self.converter.body += """iree_cc_embed_data(
%(name_block)s%(srcs_block)s%(cc_file_output_block)s%(h_file_output_block)s%(namespace_block)s%(flatten_block)s  PUBLIC\n)\n\n""" % {
    "name_block": name_block,
    "srcs_block": srcs_block,
    "cc_file_output_block": cc_file_output_block,
    "h_file_output_block": h_file_output_block,
    "namespace_block": namespace_block,
    "flatten_block": flatten_block,
    }

  def spirv_kernel_cc_library(self, name, srcs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)

    self.converter.body += """iree_spirv_kernel_cc_library(
%(name_block)s%(srcs_block)s)\n\n""" % {
    "name_block": name_block,
    "srcs_block": srcs_block,
    }

  def iree_bytecode_module(self,
                           name,
                           src,
                           translation="-iree-mlir-to-vm-bytecode-module",
                           translate_tool="//iree/tools:iree-translate",
                           cc_namespace=None):
    name_block = self._convert_name_block(name)
    src_block = self._convert_src_block(src)
    namespace_block = self._convert_cc_namespace_block(cc_namespace)
    translate_tool_block = self._convert_translate_tool_block(translate_tool)
    translation_block = self._convert_translation_block(translation)

    self.converter.body += """iree_bytecode_module(
%(name_block)s%(src_block)s%(namespace_block)s%(translate_tool_block)s%(translation_block)s  PUBLIC\n)\n\n""" % {
    "name_block": name_block,
    "src_block": src_block,
    "namespace_block": namespace_block,
    "translate_tool_block": translate_tool_block,
    "translation_block": translation_block,
    }

  def gentbl(self,
             name,
             tblgen,
             td_file,
             tbl_outs,
             td_srcs=[],
             td_includes=[],
             strip_include_prefix=None,
             test=False):
    name_block = self._convert_name_block(name)
    tblgen_block = self._convert_tblgen_block(tblgen)
    td_file_block = self._convert_td_file_block(td_file)
    outs_block = self._convert_tbl_outs_block(tbl_outs)

    self.converter.body += """iree_tablegen_library(
%(name_block)s%(td_file_block)s%(outs_block)s%(tblgen_block)s)\n\n""" % {
    "name_block": name_block,
    "td_file_block": td_file_block,
    "outs_block": outs_block,
    "tblgen_block": tblgen_block,
    }

  def iree_lit_test_suite(self, name, srcs, data, **kwargs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    data_block = self._convert_data_block(data)

    self.converter.body += ("iree_lit_test_suite(\n"
                            "%(name_block)s"
                            "%(srcs_block)s"
                            "%(data_block)s)\n\n" % {
                                "name_block": name_block,
                                "srcs_block": srcs_block,
                                "data_block": data_block,
                            })


class Converter(object):
  """Conversion state tracking and full file template substitution."""

  def __init__(self, directory_path, rel_build_file_path):
    self.body = ""
    self.directory_path = directory_path
    self.rel_build_file_path = rel_build_file_path

  def convert(self, copyright_line):
    # One `add_subdirectory(name)` per subdirectory.
    add_subdir_commands = []
    for root, dirs, file_names in os.walk(self.directory_path):
      for dir in sorted(dirs):
        if os.path.isfile(os.path.join(root, dir, "CMakeLists.txt")):
          add_subdir_commands.append("add_subdirectory(%s)" % (dir,))
      # Stop walk, only add direct subdirectories.
      break

    converted_file = self.template % {
        "copyright_line": copyright_line,
        "add_subdirectories": "\n".join(add_subdir_commands),
        "body": self.body,
    }

    # Cleanup newline characters. This is more convenient than ensuring all
    # conversions are careful with where they insert newlines.
    converted_file = converted_file.replace("\n\n\n", "\n")
    converted_file = converted_file.rstrip() + "\n"

    return converted_file

  template = textwrap.dedent("""\
    %(copyright_line)s
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

    %(add_subdirectories)s

    %(body)s""")


def GetDict(obj):
  ret = {}
  for k in dir(obj):
    if not k.startswith("_"):
      ret[k] = getattr(obj, k)
  return ret


def convert_directory_tree(root_directory_path, write_files):
  print("convert_directory_tree: %s" % (root_directory_path,))
  # Process directories starting at leaves so we can skip add_directory on
  # subdirs without a CMakeLists file.
  for root, dirs, file_names in os.walk(root_directory_path, topdown=False):
    convert_directory(root, write_files)


def convert_directory(directory_path, write_files):
  if not os.path.isdir(directory_path):
    raise FileNotFoundError("Cannot find directory '%s'" % (directory_path,))

  build_file_path = os.path.join(directory_path, "BUILD")
  cmakelists_file_path = os.path.join(directory_path, "CMakeLists.txt")

  if not os.path.isfile(build_file_path):
    # No Bazel BUILD file in this directory to convert, skip.
    return

  global repo_root
  rel_build_file_path = os.path.relpath(build_file_path, repo_root)
  rel_cmakelists_file_path = os.path.relpath(cmakelists_file_path, repo_root)
  print("Converting %s to %s" % (rel_build_file_path, rel_cmakelists_file_path))

  cmake_file_exists = os.path.isfile(cmakelists_file_path)
  copyright_line = "# Copyright %s Google LLC" % (datetime.date.today().year,)
  write_allowed = write_files
  if cmake_file_exists:
    with open(cmakelists_file_path) as f:
      for i, line in enumerate(f):
        if line.startswith("# Copyright"):
          copyright_line = line.rstrip()
        if EDIT_BLOCKING_PATTERN.search(line):
          print(
              "  %(path)s already exists, and line %(index)d: '%(line)s' prevents edits. Falling back to preview"
              % {
                  "path": rel_cmakelists_file_path,
                  "index": i + 1,
                  "line": line.strip()
              })
          write_allowed = False

  if write_allowed:
    # TODO(scotttodd): Attempt to merge instead of overwrite?
    #   Existing CMakeLists.txt may have special logic that should be preserved
    if cmake_file_exists:
      print("  %s already exists, overwriting" % (rel_cmakelists_file_path,))
    else:
      print("  %s does not exist yet, creating" % (rel_cmakelists_file_path,))
  print("")

  with open(build_file_path, "rt") as build_file:
    build_file_code = compile(build_file.read(), build_file_path, "exec")
    converter = Converter(directory_path, rel_build_file_path)
    try:
      exec(build_file_code, GetDict(BuildFileFunctions(converter)))
      converted_text = converter.convert(copyright_line)
      if write_allowed:
        with open(cmakelists_file_path, "wt") as cmakelists_file:
          cmakelists_file.write(converted_text)
      else:
        print(converted_text)
    except NameError as e:
      print(
          "Failed to convert %s. Missing a rule handler in bazel_to_cmake.py?" %
          (rel_build_file_path))
      print("  Reason: `%s: %s`" % (type(e).__name__, e))
    except KeyError as e:
      print(
          "Failed to convert %s. Missing a conversion in bazel_to_cmake_targets.py?"
          % (rel_build_file_path))
      print("  Reason: `%s: %s`" % (type(e).__name__, e))


def main(args):
  """Runs Bazel to CMake conversion."""
  global repo_root

  write_files = not args.preview

  if args.root_dir:
    convert_directory_tree(os.path.join(repo_root, args.root_dir), write_files)
  elif args.dir:
    convert_directory(os.path.join(repo_root, args.dir), write_files)


if __name__ == "__main__":
  setup_environment()
  main(parse_arguments())
