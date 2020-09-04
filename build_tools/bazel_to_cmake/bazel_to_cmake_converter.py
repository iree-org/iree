# Lint as: python3
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
"""Converter class for converting Bazel BUILD files to CMakeLists.txt files.

See bazel_to_cmake.py for usage.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=exec-used

import itertools
import textwrap

import bazel_to_cmake_targets


def _expand_cmake_var(var):
  return "${" + var + "}"


class BuildFileFunctions(object):
  """Object passed to `exec` that has handlers for BUILD file functions."""

  def __init__(self, converter):
    self.converter = converter
    # TODO(gcmn): Do this in a less hard-coded way
    self.PLATFORM_VULKAN_DEPS = []
    self.PLATFORM_VULKAN_TEST_DEPS = ["//iree/testing:gtest_main"]
    self.FLATBUFFER_SUPPORTS_REFLECTIONS = False
    self.PLATFORM_VULKAN_LOADER_COPTS = []
    self.IREE_DRIVER_MODULES = [
        "//iree/hal/vmla:vmla_driver_module",
        "//iree/hal/vulkan:vulkan_driver_module",
        "//iree/hal/llvmjit:llvmjit_driver_module",
    ]

  # ------------------------------------------------------------------------- #
  # Conversion utilities, written to reduce boilerplate and allow for reuse   #
  # between similar rule conversions (e.g. cc_library and cc_binary).         #
  # ------------------------------------------------------------------------- #

  def _convert_string_arg_block(self, name, value):
    #  NAME
    #    value
    return f"  {name}\n    {value}\n"

  def _convert_name_block(self, name):
    #  NAME
    #    rule_name
    return f"  NAME\n    {name}\n"

  def _convert_out_block(self, out):
    #  OUT
    #    out_name
    return f"  OUT\n    {out}\n"

  def _convert_cc_namespace_block(self, cc_namespace):
    #  CC_NAMESPACE
    #    "cc_namespace"
    if not cc_namespace:
      return ""
    return f'  CC_NAMESPACE\n    "{cc_namespace}"\n'

  def _convert_cpp_namespace_block(self, cpp_namespace):
    #  CPP_NAMESPACE
    #    "cpp_namespace"
    if not cpp_namespace:
      return ""
    return f'  CPP_NAMESPACE\n    "{cpp_namespace}"\n'

  def _convert_string_list_block(self, name, values):
    # Note this deliberately distinguishes between an empty list (argument
    # explicitly specified) and None (argument left as default).
    if values is None:
      return ""
    values_list = "\n".join([f'    "{v}"' for v in values])
    return f"  {name}\n{values_list}\n"

  def _convert_translate_tool_block(self, translate_tool):
    if translate_tool and translate_tool != "//iree/tools:iree-translate":
      # Bazel `//iree/base`     -> CMake `iree::base`
      # Bazel `//iree/base:api` -> CMake `iree::base::api`
      translate_tool = translate_tool.replace("//iree", "iree")  # iree/base:api
      translate_tool = translate_tool.replace(":", "_")  # iree/base::api
      translate_tool = translate_tool.replace("/", "_")  # iree::base::api
      return f"  TRANSLATE_TOOL\n    {translate_tool}\n"
    else:
      return ""

  def _convert_option_block(self, option, option_value):
    if option_value:
      # Note: this is a truthiness check as well as an existence check, i.e.
      # Bazel `testonly = False` will be handled correctly by this condition.
      return f"  {option}\n"
    else:
      return ""

  def _convert_alwayslink_block(self, alwayslink):
    return self._convert_option_block("ALWAYSLINK", alwayslink)

  def _convert_testonly_block(self, testonly):
    return self._convert_option_block("TESTONLY", testonly)

  def _convert_flatten_block(self, flatten):
    return self._convert_option_block("FLATTEN", flatten)

  def _convert_file_list_block(self, list_name, files):
    #  list_name
    #    "file_1.h"
    #    "file_2.h"
    #    "file_3.h"
    if not files:
      return ""
    files_list = "\n".join([f'    "{file}"' for file in sorted(files)])
    return f"  {list_name}\n{files_list}\n"

  def _convert_hdrs_block(self, hdrs):
    return self._convert_file_list_block("HDRS", hdrs)

  def _convert_textual_hdrs_block(self, textual_hdrs):
    return self._convert_file_list_block("TEXTUAL_HDRS", textual_hdrs)

  def _convert_srcs_block(self, srcs):
    if not srcs:
      return ""
    generated_srcs = [src for src in srcs if src.startswith(":")]
    srcs = [src for src in srcs if src not in generated_srcs]
    sets = []
    if srcs:
      sets.append(self._convert_file_list_block("SRCS", srcs))
    if generated_srcs:
      sets.append(
          self._convert_file_list_block("GENERATED_SRCS",
                                        [src[1:] for src in generated_srcs]))
    return "\n".join(sets)

  def _convert_src_block(self, src):
    return f'  SRC\n    "{src}"\n'

  def _convert_cc_file_output_block(self, cc_file_output):
    return f'  CC_FILE_OUTPUT\n    "{cc_file_output}"\n'

  def _convert_h_file_output_block(self, h_file_output):
    return f'  H_FILE_OUTPUT\n    "{h_file_output}"\n'

  def _convert_td_file_block(self, td_file):
    if td_file.startswith("//iree"):
      # Bazel `//iree/dir/td_file.td`
      # -> CMake `${IREE_ROOT_DIR}/iree/dir/td_file.td
      # Bazel `//iree/dir/IR:td_file.td`
      # -> CMake `${IREE_ROOT_DIR}/iree/dir/IR/td_file.td
      td_file = td_file.replace("//iree", "${IREE_ROOT_DIR}/iree")
      td_file = td_file.replace(":", "/")
    return f'  TD_FILE\n    "{td_file}"\n'

  def _convert_tbl_outs_block(self, tbl_outs):
    outs_list = "\n".join([f"    {flag} {value}" for flag, value in tbl_outs])
    return f"  OUTS\n{outs_list}\n"

  def _convert_tblgen_block(self, tblgen):
    if tblgen.endswith("iree-tblgen"):
      return "  TBLGEN\n    IREE\n"
    else:
      return ""

  def _convert_target(self, target):
    if target.startswith(":") and target.endswith(("_gen", "Gen")):
      # Files created by gentbl have to be included as source and header files
      # and not as a dependency. Adding these targets to the dependencies list,
      # results in linkage failures if the library including the gentbl dep is
      # marked as ALWAYSLINK.
      # This drops deps in the local namespace ending with '_gen' and 'Gen'
      target = [""]
    elif not target.startswith(("//bindings", "//experimental", "//iree", ":")):
      # External target, call helper method for special case handling.
      target = bazel_to_cmake_targets.convert_external_target(target)
    else:
      # Bazel `:api`            -> CMake `::api`
      # Bazel `//iree/base`     -> CMake `iree::base`
      # Bazel `//iree/base:api` -> CMake `iree::base::api`
      target = target.replace("//bindings", "bindings")  # bindings:api
      # Support for experimental targets is best effort with no guarantees.
      target = target.replace("//experimental",
                              "experimental")  # experimental:api
      target = target.replace("//iree", "iree")  # iree/base:api
      target = target.replace(":", "::")  # iree/base::api or ::api
      target = target.replace("/", "::")  # iree::base::api
      target = [target]
    return target

  def _convert_target_list_block(self, list_name, targets):
    if not targets:
      return ""

    #  DEPS
    #    package1::target1
    #    package1::target2
    #    package2::target
    targets = [self._convert_target(t) for t in targets]
    # Flatten lists
    targets = list(itertools.chain.from_iterable(targets))
    # Remove duplicates
    targets = set(targets)
    # Remove Falsey (None and empty string) values
    targets = filter(None, targets)
    # Sort the targets and convert to a list
    targets = sorted(targets)
    target_list_string = "\n".join([f"    {target}" for target in targets])
    return f"  {list_name}\n{target_list_string}\n"

  def _convert_data_block(self, data):
    return self._convert_target_list_block("DATA", data)

  def _convert_deps_block(self, deps):
    return self._convert_target_list_block("DEPS", deps)

  def _convert_flatc_args_block(self, flatc_args):
    if not flatc_args:
      return ""
    flatc_args = "\n".join([f'    "{flatc_arg}"' for flatc_arg in flatc_args])
    return f"  FLATC_ARGS\n{flatc_args}\n"

  def _convert_flatcc_args_block(self, flatcc_args):
    if not flatcc_args:
      return ""
    flatcc_args = "\n".join(
        [f'    "{flatcc_arg}"' for flatcc_arg in flatcc_args])
    return f"  FLATCC_ARGS\n{flatcc_args}\n"

  def _convert_unimplemented_function(self, function, details=""):
    message = f"Unimplemented {function}: {details}"
    if not self.converter.first_error:
      self.converter.first_error = NotImplementedError(message)
    # Avoid submitting the raw results from non-strict runs. These are still
    # useful but are generally not safe to submit as-is. An upstream check
    # prevents changes with this phrase from being submitted.
    # Written as separate literals to avoid the check triggering here.
    submit_blocker = "DO" + " NOT" + " SUBMIT."
    self.converter.body += f"# {submit_blocker} {message}\n"

  # ------------------------------------------------------------------------- #
  # Function handlers that convert BUILD definitions to CMake definitions.    #
  #                                                                           #
  # Names and signatures must match 1:1 with those expected in BUILD files.   #
  # Each function that may be found in a BUILD file must be listed here.      #
  # ------------------------------------------------------------------------- #

  def load(self, *args, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def package(self, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def iree_build_test(self, **kwargs):
    pass

  def test_suite(self, **kwargs):
    # No CMake equivalent, ignore.
    pass

  def filegroup(self, name, **kwargs):
    # Not implemented yet. Might be a no-op, or may want to evaluate the srcs
    # attribute and pass them along to any targets that depend on the filegroup.
    # Cross-package dependencies and complicated globs could be hard to handle.

    # We have a bunch of filegroups that just contain TD files. CMake doesn't
    # model this at all, so we'll just hardcode this special case.
    # TODO(gcmn): Handle this robustly
    if name == "td_files":
      return

    self._convert_unimplemented_function("filegroup", name)

  def sh_binary(self, name, **kwargs):
    self._convert_unimplemented_function("sh_binary", name)

  def exports_files(self, *args, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def glob(self, include, exclude=None, exclude_directories=1):
    if exclude_directories != 1:
      self._convert_unimplemented_function("glob", "with exclude_directories")
    if exclude is None:
      exclude = []

    glob_vars = []
    for pattern in include:
      if "**" in pattern:
        # bazel's glob has some specific restrictions about crossing package
        # boundaries. We have no uses of recursive globs. Rather than try to
        # emulate them or silently give different behavior, just error out.
        # See https://docs.bazel.build/versions/master/be/functions.html#glob
        raise NotImplementedError("Recursive globs not supported")
      # Bazel `*.mlir` glob -> CMake Variable `_GLOB_X_MLIR`
      var = "_GLOB_" + pattern.replace("*", "X").replace(".", "_").upper()
      glob_vars.append(var)
      self.converter.body += (
          f"file(GLOB {var} LIST_DIRECTORIES false"
          f" RELATIVE {_expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
          f" CONFIGURE_DEPENDS {pattern})\n")
    for pattern in exclude:
      if "**" in pattern:
        raise NotImplementedError("Recursive globs not supported")
      exclude_var = ("_GLOB_" +
                     pattern.replace("*", "X").replace(".", "_").upper())
      self.converter.body += (
          f"file(GLOB {exclude_var} LIST_DIRECTORIES false"
          f" RELATIVE {_expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
          f" CONFIGURE_DEPENDS {pattern})\n")
      for glob_var in glob_vars:
        self.converter.body += (
            f"list(REMOVE_ITEM {glob_var} {_expand_cmake_var(exclude_var)})\n")
    return [_expand_cmake_var(var) for var in glob_vars]

  # TODO(gcmn) implement these types of functions in a less hard-coded way
  def platform_trampoline_deps(self, basename, path="base"):
    return [f"//iree/{path}/internal:{basename}_internal"]

  def select(self, d):
    self._convert_unimplemented_function("select", str(d))
    return d["//conditions:default"]

  def config_setting(self, **kwargs):
    # No mapping to CMake, ignore.
    pass

  def cc_library(self,
                 name,
                 hdrs=None,
                 textual_hdrs=None,
                 srcs=None,
                 data=None,
                 deps=None,
                 alwayslink=False,
                 testonly=False,
                 linkopts=None,
                 **kwargs):
    if linkopts:
      self._convert_unimplemented_function("linkopts")
    name_block = self._convert_name_block(name)
    hdrs_block = self._convert_hdrs_block(hdrs)
    textual_hdrs_block = self._convert_textual_hdrs_block(textual_hdrs)
    srcs_block = self._convert_srcs_block(srcs)
    data_block = self._convert_data_block(data)
    deps_block = self._convert_deps_block(deps)
    alwayslink_block = self._convert_alwayslink_block(alwayslink)
    testonly_block = self._convert_testonly_block(testonly)

    self.converter.body += (f"iree_cc_library(\n"
                            f"{name_block}"
                            f"{hdrs_block}"
                            f"{textual_hdrs_block}"
                            f"{srcs_block}"
                            f"{data_block}"
                            f"{deps_block}"
                            f"{alwayslink_block}"
                            f"{testonly_block}"
                            f"  PUBLIC\n)\n\n")

  def cc_test(self,
              name,
              hdrs=None,
              srcs=None,
              data=None,
              deps=None,
              tags=None,
              **kwargs):
    name_block = self._convert_name_block(name)
    hdrs_block = self._convert_hdrs_block(hdrs)
    srcs_block = self._convert_srcs_block(srcs)
    data_block = self._convert_data_block(data)
    deps_block = self._convert_deps_block(deps)
    labels_block = self._convert_string_list_block("LABELS", tags)

    self.converter.body += (f"iree_cc_test(\n"
                            f"{name_block}"
                            f"{hdrs_block}"
                            f"{srcs_block}"
                            f"{data_block}"
                            f"{deps_block}"
                            f"{labels_block}"
                            f")\n\n")

  def cc_binary(self,
                name,
                srcs=None,
                data=None,
                deps=None,
                linkopts=None,
                testonly=False,
                **kwargs):
    if linkopts:
      self._convert_unimplemented_function("linkopts")
    name_block = self._convert_name_block(name)
    out_block = self._convert_out_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    data_block = self._convert_data_block(data)
    deps_block = self._convert_deps_block(deps)
    testonly_block = self._convert_testonly_block(testonly)

    self.converter.body += (f"iree_cc_binary(\n"
                            f"{name_block}"
                            f"{out_block}"
                            f"{srcs_block}"
                            f"{data_block}"
                            f"{deps_block}"
                            f"{testonly_block}"
                            f")\n\n")

  # Effectively an alias in IREE code.
  iree_cc_binary = cc_binary

  def cc_embed_data(self,
                    name,
                    srcs,
                    cc_file_output,
                    h_file_output,
                    testonly=False,
                    cpp_namespace=None,
                    strip_prefix=None,
                    flatten=False,
                    identifier=None,
                    **kwargs):
    if identifier:
      self._convert_unimplemented_function("cc_embed_data",
                                           name + " has identifier")
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    cc_file_output_block = self._convert_cc_file_output_block(cc_file_output)
    h_file_output_block = self._convert_h_file_output_block(h_file_output)
    testonly_block = self._convert_testonly_block(testonly)
    namespace_block = self._convert_cpp_namespace_block(cpp_namespace)
    flatten_block = self._convert_flatten_block(flatten)

    self.converter.body += (f"iree_cc_embed_data(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{cc_file_output_block}"
                            f"{h_file_output_block}"
                            f"{testonly_block}"
                            f"{namespace_block}"
                            f"{flatten_block}"
                            f"  PUBLIC\n)\n\n")

  def spirv_kernel_cc_library(self, name, srcs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)

    self.converter.body += (f"iree_spirv_kernel_cc_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f")\n\n")

  def iree_bytecode_module(self,
                           name,
                           src,
                           flags=["-iree-mlir-to-vm-bytecode-module"],
                           translate_tool="//iree/tools:iree-translate",
                           cc_namespace=None):
    name_block = self._convert_name_block(name)
    src_block = self._convert_src_block(src)
    namespace_block = self._convert_cc_namespace_block(cc_namespace)
    translate_tool_block = self._convert_translate_tool_block(translate_tool)
    flags_block = self._convert_string_list_block("FLAGS", flags)

    self.converter.body += (f"iree_bytecode_module(\n"
                            f"{name_block}"
                            f"{src_block}"
                            f"{namespace_block}"
                            f"{translate_tool_block}"
                            f"{flags_block}"
                            f"  PUBLIC\n)\n\n")

  def iree_flatbuffer_c_library(self, name, srcs, flatcc_args=None):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    flatcc_args_block = self._convert_flatcc_args_block(flatcc_args)

    self.converter.body += (f"flatbuffer_c_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{flatcc_args_block}"
                            f"  PUBLIC\n)\n\n")

  def iree_flatbuffer_cc_library(self, name, srcs, flatc_args=None):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    flatc_args_block = self._convert_flatc_args_block(flatc_args)

    self.converter.body += (f"flatbuffer_cc_library(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{flatc_args_block}"
                            f"  PUBLIC\n)\n\n")

  def gentbl(self,
             name,
             tblgen,
             td_file,
             tbl_outs,
             td_srcs=None,
             td_includes=None,
             strip_include_prefix=None,
             test=False):
    name_block = self._convert_name_block(name)
    tblgen_block = self._convert_tblgen_block(tblgen)
    td_file_block = self._convert_td_file_block(td_file)
    outs_block = self._convert_tbl_outs_block(tbl_outs)

    self.converter.body += (f"iree_tablegen_library(\n"
                            f"{name_block}"
                            f"{td_file_block}"
                            f"{outs_block}"
                            f"{tblgen_block}"
                            f")\n\n")

  def iree_tablegen_doc(self,
                        name,
                        tblgen,
                        td_file,
                        tbl_outs,
                        td_srcs=None,
                        td_includes=None,
                        strip_include_prefix=None):
    name_block = self._convert_name_block(name)
    tblgen_block = self._convert_tblgen_block(tblgen)
    td_file_block = self._convert_td_file_block(td_file)
    outs_block = self._convert_tbl_outs_block(tbl_outs)

    self.converter.body += (f"iree_tablegen_doc(\n"
                            f"{name_block}"
                            f"{td_file_block}"
                            f"{outs_block}"
                            f"{tblgen_block}"
                            f")\n\n")

  def iree_lit_test_suite(self, name, srcs, data, tags=None, **kwargs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    data_block = self._convert_data_block(data)
    labels_block = self._convert_string_list_block("LABELS", tags)

    self.converter.body += (f"iree_lit_test_suite(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{data_block}"
                            f"{labels_block}"
                            f")\n\n")

  def iree_check_single_backend_test_suite(self,
                                           name,
                                           srcs,
                                           target_backend,
                                           driver,
                                           compiler_flags=None,
                                           target_backends_and_drivers=None,
                                           runner_args=None,
                                           tags=None,
                                           **kwargs):
    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    target_backend_block = self._convert_string_arg_block(
        "TARGET_BACKEND", target_backend)
    driver_block = self._convert_string_arg_block("DRIVER", driver)
    compiler_flags_block = self._convert_string_list_block(
        "COMPILER_FLAGS", compiler_flags)
    runner_args_block = self._convert_string_list_block("RUNNER_ARGS",
                                                        runner_args)
    labels_block = self._convert_string_list_block("LABELS", tags)

    self.converter.body += (f"iree_check_single_backend_test_suite(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{target_backend_block}"
                            f"{driver_block}"
                            f"{compiler_flags_block}"
                            f"{runner_args_block}"
                            f"{labels_block}"
                            f")\n\n")

  def iree_check_test_suite(self,
                            name,
                            srcs,
                            target_backends_and_drivers=None,
                            compiler_flags=None,
                            runner_args=None,
                            tags=None,
                            **kwargs):
    target_backends = None
    drivers = None
    if target_backends_and_drivers is not None:
      target_backends = [it[0] for it in target_backends_and_drivers]
      drivers = [it[1] for it in target_backends_and_drivers]

    name_block = self._convert_name_block(name)
    srcs_block = self._convert_srcs_block(srcs)
    target_backends_block = self._convert_string_list_block(
        "TARGET_BACKENDS", target_backends)
    drivers_block = self._convert_string_list_block("DRIVERS", drivers)
    compiler_flags_block = self._convert_string_list_block(
        "COMPILER_FLAGS", compiler_flags)
    runner_args_block = self._convert_string_list_block("RUNNER_ARGS",
                                                        runner_args)
    labels_block = self._convert_string_list_block("LABELS", tags)

    self.converter.body += (f"iree_check_test_suite(\n"
                            f"{name_block}"
                            f"{srcs_block}"
                            f"{target_backends_block}"
                            f"{drivers_block}"
                            f"{compiler_flags_block}"
                            f"{runner_args_block}"
                            f"{labels_block}"
                            f")\n\n")

  def iree_cmake_extra_content(self, content, inline=False):
    if inline:
      self.converter.body += (f"\n{content}\n")
    else:
      self.converter.header += (f"\n{content}\n")


class Converter(object):
  """Conversion state tracking and full file template substitution."""

  def __init__(self):
    # Header appears after the license block but before `iree_add_all_subdirs`.
    self.header = ""
    # Body appears after `iree_add_all_subdirs`.
    self.body = ""

    self.first_error = None

  def convert(self, copyright_line):
    converted_content = (f"{copyright_line}\n"
                         f"{self.apache_license}\n\n"
                         f"{self.header}\n\n"
                         f"iree_add_all_subdirs()\n\n"
                         f"{self.body}")

    # Cleanup newline characters. This is more convenient than ensuring all
    # conversions are careful with where they insert newlines.
    converted_content = converted_content.replace("\n\n\n", "\n")
    converted_content = converted_content.rstrip() + "\n"

    return converted_content

  apache_license = textwrap.dedent("""\
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
    # limitations under the License.""")


def GetDict(obj):
  ret = {}
  for k in dir(obj):
    if not k.startswith("_"):
      ret[k] = getattr(obj, k)
  return ret


def convert_build_file(build_file_code,
                       copyright_line,
                       allow_partial_conversion=False):
  converter = Converter()
  exec(build_file_code, GetDict(BuildFileFunctions(converter)))
  converted_text = converter.convert(copyright_line)
  if not allow_partial_conversion and converter.first_error:
    raise converter.first_error  # pylint: disable=raising-bad-type
  return converted_text
