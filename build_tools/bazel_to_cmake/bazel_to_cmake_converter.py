# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Converter class for converting Bazel BUILD files to CMakeLists.txt files.

See bazel_to_cmake.py for usage.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=exec-used

import itertools
import re
import os

import bazel_to_cmake_targets


class BuildFileFunctions(object):
    """Object passed to `exec` that has handlers for BUILD file functions."""

    def __init__(
        self,
        *,
        converter: "Converter",
        targets: bazel_to_cmake_targets.TargetConverter,
        build_dir: str,
    ):
        self._converter = converter
        self._targets = targets
        self._build_dir = build_dir
        self._custom_initialize()

    def _custom_initialize(self):
        pass

    # ------------------------------------------------------------------------- #
    # Conversion utilities, written to reduce boilerplate and allow for reuse   #
    # between similar rule conversions (e.g. cc_library and cc_binary).         #
    # ------------------------------------------------------------------------- #

    def _expand_cmake_var(self, var):
        return "${" + var + "}"

    def _convert_string_arg_block(self, name, value, quote=True):
        #  NAME
        #    "value"
        if value is None:
            return ""
        if quote:
            return f'  {name}\n    "{value}"\n'
        else:
            return f"  {name}\n    {value}\n"

    # Match Bazel's timeout values
    # https://docs.bazel.build/versions/main/test-encyclopedia.html
    _timeout_map = {
        "short": 60,
        "moderate": 300,
        "long": 900,
        "eternal": 3600,
    }

    def _should_skip_target(self, tags=None, **kwargs):
        if tags and "skip-bazel_to_cmake" in tags:
            return True
        return False

    def _convert_timeout_arg_block(self, name, value):
        if value is None:
            return ""
        value = self._timeout_map[value]
        return f"  {name}\n    {value}\n"

    def _convert_string_list_block(self, name, values, quote=True, sort=False):
        # Note this deliberately distinguishes between an empty list (argument
        # explicitly specified) and None (argument left as default).
        if values is None:
            return ""

        if sort:
            values = sorted(values)

        if quote:
            values_list = "\n".join([f'    "{v}"' for v in values])
        else:
            values_list = "\n".join([f"    {v}" for v in values])

        return f"  {name}\n{values_list}\n"

    def _convert_option_block(self, option, option_value):
        if option_value:
            # Note: this is a truthiness check as well as an existence check, e.g.
            # Bazel `testonly = False` will be handled correctly by this condition.
            return f"  {option}\n"
        else:
            return ""

    def _convert_target_block(self, name, target):
        if target is None:
            return ""

        # Convert the target name from its Bazel name to the corresponding CMake name.
        # The specific conversion pattern depends on the target location. In general,
        # Bazel targets are fully qualified and use slashes as delimiters, while
        # targets in CMake are rooted on subtrees and use _ (with :: aliases).
        cmake_aliases = self._targets.convert_target(target)
        if len(cmake_aliases) != 1:
            raise ValueError(
                f"Expected a CMake alias from {target}. Got {cmake_aliases}"
            )
        target = cmake_aliases[0]
        # Replace aliased :: target names with their explicit _ names.
        target = target.replace("::", "_")
        return self._convert_string_arg_block(name, target, quote=False)

    def _filegroup_dep_filename(self, src):
        return f"{src}.stamp"

    def _normalize_label(self, src):
        """
        Convert label to file path suitable for CMake to use as a dependency.
        """

        # Bazel allows srcs to reference targets in the current package (leading
        # ':') or in other packages (leading '//'). We map that to paths by:
        # - dropping any leading ':' as in:
        #      ':generated.c' -> 'generated.c'
        # - replacing any leading '//' by '${CMAKE_SOURCE_DIR}/' or
        #   '${CMAKE_BINARY_DIR}/' and any internal ':' by '/', as in:
        #      '//path/to/package:source.c'
        #      -> '${CMAKE_SOURCE_DIR}/path/to/package/source.c'
        #      '//path/to/package:generated.c'
        #      -> '${CMAKE_BINARY_DIR}/path/to/package/generated.c'
        pkg_root_relative_label = src.startswith("//")
        src = src.lstrip("/").lstrip(":").replace(":", "/")
        if not pkg_root_relative_label:
            return src
        elif os.path.exists(os.path.join(self._build_dir, src)):
            return f"${{CMAKE_SOURCE_DIR}}/{src}"
        else:
            return f"${{CMAKE_BINARY_DIR}}/{src}"

    def _convert_srcs_block(self, srcs, is_generated=False, block_name="SRCS"):
        if not srcs:
            return ""

        srcs = [
            self._normalize_label(s)
            if s.startswith("$") or os.path.splitext(s)[1]
            else self._filegroup_dep_filename(self._normalize_label(s))
            for s in srcs
        ]

        return self._convert_string_list_block(block_name, srcs, sort=True)

    def _convert_td_file_block(self, td_file):
        if td_file.startswith("//iree"):
            # TODO: This should be generalized for out of tree.
            # Bazel `//iree/dir/td_file.td`
            # -> CMake `${IREE_ROOT_DIR}/iree/dir/td_file.td
            # Bazel `//iree/dir/IR:td_file.td`
            # -> CMake `${IREE_ROOT_DIR}/iree/dir/IR/td_file.td
            td_file = td_file.replace("//iree", "${IREE_ROOT_DIR}/iree")
            td_file = td_file.replace(":", "/")
        return self._convert_string_arg_block("TD_FILE", td_file)

    def _convert_tbl_outs_block(self, tbl_outs):
        outs_list = "\n".join(
            [f"    {' '.join(flags)} {value}" for flags, value in tbl_outs]
        )
        return f"  OUTS\n{outs_list}\n"

    def _convert_tblgen_block(self, tblgen):
        if tblgen.endswith("iree-tblgen"):
            return "  TBLGEN\n    IREE\n"
        else:
            return ""

    def _convert_target(self, target):
        """Returns a list of targets that correspond to the specified Bazel target.
        Note that this must be a list because some targets have a one to many mapping.
        """
        return self._targets.convert_target(target)

    def _convert_single_target(self, target):
        replacement_targets = self._convert_target(target)
        if len(replacement_targets) != 1:
            raise RuntimeError(
                f"Expected single target replacement for {target},"
                f" but got multiple: {replacement_targets}"
            )
        return replacement_targets[0]

    def _convert_single_target_block(self, name, target):
        mapped_target = self._convert_single_target(target)
        return self._convert_string_arg_block(name, mapped_target, quote=False)

    def _convert_target_list_block(self, list_name, targets):
        if targets is None:
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

        return self._convert_string_list_block(
            list_name, targets, sort=True, quote=False
        )

    def _convert_includes_block(self, includes):
        if not includes:
            return ""
        dirs = []
        for include in includes:
            dirs.append(
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/%s>" % (include,)
            )
            dirs.append(
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/%s>" % (include,)
            )
        return self._convert_string_list_block("INCLUDES", dirs, sort=False, quote=True)

    def _convert_unimplemented_function(self, function, details=""):
        message = f"Unimplemented {function}: {details}"
        if not self._converter.first_error:
            self._converter.first_error = NotImplementedError(message)
        # Avoid submitting the raw results from non-strict runs. These are still
        # useful but are generally not safe to submit as-is. An upstream check
        # prevents changes with this phrase from being submitted.
        # Written as separate literals to avoid the check triggering here.
        submit_blocker = "DO" + " NOT" + " SUBMIT."
        self._converter.body += f"# {submit_blocker} {message}\n"

    # ------------------------------------------------------------------------- #
    # Function handlers that convert BUILD definitions to CMake definitions.    #
    #                                                                           #
    # Names and signatures must match 1:1 with those expected in BUILD files    #
    # except that default values for optional arguments should generally be     #
    # `None` so we don't set them unnecessarily in the CMakeLists.txt files.    #
    # Each function that may be found in a BUILD file must be listed here.      #
    # ------------------------------------------------------------------------- #

    # Functions with no mapping to CMake. Just ignore these.
    def alias(self, *args, **kwargs):
        pass

    def bool_flag(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def package(self, **kwargs):
        pass

    def iree_build_test(self, **kwargs):
        pass

    def test_suite(self, **kwargs):
        pass

    def config_setting(self, **kwargs):
        pass

    def exports_files(self, *args, **kwargs):
        pass

    def iree_td_library(self, *args, **kwargs):
        pass

    # Technically we could do something with a CMake equivalent but we have no use
    # case.
    def py_binary(self, *args, **kwargs):
        pass

    def filegroup(self, name, srcs, **kwargs):
        if not srcs:
            return

        # Converting a dependency on a filegroup requires either using the
        # transitive dependency to the actual file or creating a similar
        # abstraction in CMake.
        #
        # One way of doing the transitive dependency is peeking in the build
        # file that defines a given filegroup but goes against the current
        # design where each build file is processed independently.
        #
        # Alternatively, the build file that defines a filegroup could set a
        # variable with the list of all the files in the filegroup which the
        # CMakeLists.txt corresponding to the using build file would use.
        # However that requires the variable to be defined before the
        # add_directory() for the corresponding using CMakeLists.txt which is
        # not a given.
        #
        # Instead, we generate a custom command that creates a stamp file that
        # acts as an abstraction to the filegroup. The using CMakeLists.txt
        # then creates a file dependency on that stamp file. We also need a
        # custom target in the same CMakeLists.txt to ensure a rule for the
        # custom command is actually created as per add_custom_command
        # documentation.
        depends_block = self._convert_srcs_block(srcs, block_name="DEPENDS")
        stamp_file = self._filegroup_dep_filename(name)
        self._converter.body += (
            f"add_custom_command(OUTPUT {stamp_file}\n"
            f"    COMMAND touch {stamp_file}\n"
            f"{depends_block}"
            f")\n\n"
            f"add_custom_target({name}\n"
            f"    DEPENDS {stamp_file}\n"
            f")\n\n"
        )

    def sh_binary(self, name, **kwargs):
        if self._should_skip_target(**kwargs):
            return
        self._convert_unimplemented_function("sh_binary", name)

    def enforce_glob(self, files, **kwargs):
        return files

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
            self._converter.body += (
                f"file(GLOB {var} LIST_DIRECTORIES false"
                f" RELATIVE {self._expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
                f" CONFIGURE_DEPENDS {pattern})\n"
            )
        for pattern in exclude:
            if "**" in pattern:
                raise NotImplementedError("Recursive globs not supported")
            exclude_var = "_GLOB_" + pattern.replace("*", "X").replace(".", "_").upper()
            self._converter.body += (
                f"file(GLOB {exclude_var} LIST_DIRECTORIES false"
                f" RELATIVE {self._expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
                f" CONFIGURE_DEPENDS {pattern})\n"
            )
            for glob_var in glob_vars:
                self._converter.body += f"list(REMOVE_ITEM {glob_var} {self._expand_cmake_var(exclude_var)})\n"
        return [self._expand_cmake_var(var) for var in glob_vars]

    # TODO(gcmn) implement these types of functions in a less hard-coded way
    def platform_trampoline_deps(self, basename, path="base"):
        return [f"//{path}/internal:{basename}_internal"]

    def select(self, d):
        self._convert_unimplemented_function("select", str(d))
        return d["//conditions:default"]

    def defaulting_select(self, selector):
        """Defined in build_defs.oss.bzl as a scoped alternative to select."""
        default_value = selector.get("//conditions:default")
        if default_value is None:
            raise ValueError("bazel_to_cmake can only convert selects with a default")
        return default_value

    def cc_library(
        self,
        name,
        hdrs=None,
        textual_hdrs=None,
        srcs=None,
        copts=None,
        defines=None,
        data=None,
        deps=None,
        testonly=None,
        linkopts=None,
        includes=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        if linkopts:
            self._convert_unimplemented_function("linkopts")
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        hdrs_block = self._convert_string_list_block("HDRS", hdrs, sort=True)
        textual_hdrs_block = self._convert_string_list_block(
            "TEXTUAL_HDRS", textual_hdrs, sort=True
        )
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        includes_block = self._convert_includes_block(includes)

        self._converter.body += (
            f"iree_cc_library(\n"
            f"{name_block}"
            f"{copts_block}"
            f"{hdrs_block}"
            f"{textual_hdrs_block}"
            f"{srcs_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{defines_block}"
            f"{testonly_block}"
            f"{includes_block}"
            f"  PUBLIC\n)\n\n"
        )

    def iree_compiler_register_plugin(self, plugin_id, target):
        plugin_id_block = self._convert_string_arg_block(
            "PLUGIN_ID", plugin_id, quote=False
        )
        target_block = self._convert_single_target_block("TARGET", target)
        self._converter.body += (
            f"iree_compiler_register_plugin(\n"
            f"{plugin_id_block}"
            f"{target_block}"
            f")\n\n"
        )

    def cc_test(
        self,
        name,
        hdrs=None,
        srcs=None,
        copts=None,
        defines=None,
        data=None,
        deps=None,
        timeout=None,
        args=None,
        tags=None,
        includes=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        hdrs_block = self._convert_string_list_block("HDRS", hdrs, sort=True)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        args_block = self._convert_string_list_block("ARGS", args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)
        includes_block = self._convert_includes_block(includes)

        self._converter.body += (
            f"iree_cc_test(\n"
            f"{name_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{args_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f"{includes_block}"
            f")\n\n"
        )

    def cc_binary(
        self,
        name,
        srcs=None,
        data=None,
        deps=None,
        copts=None,
        defines=None,
        linkopts=None,
        testonly=None,
        includes=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        if linkopts:
            self._convert_unimplemented_function("linkopts")
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        includes_block = self._convert_includes_block(includes)

        self._converter.body += (
            f"iree_cc_binary(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"{includes_block}"
            f")\n\n"
        )

    def c_embed_data(
        self,
        name,
        srcs,
        c_file_output,
        h_file_output,
        testonly=None,
        strip_prefix=None,
        flatten=None,
        identifier=None,
        deps=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        c_file_output_block = self._convert_string_arg_block(
            "C_FILE_OUTPUT", c_file_output
        )
        h_file_output_block = self._convert_string_arg_block(
            "H_FILE_OUTPUT", h_file_output
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        flatten_block = self._convert_option_block("FLATTEN", flatten)
        deps_block = self._convert_target_list_block("DEPS", deps)

        self._converter.body += (
            f"iree_c_embed_data(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{deps_block}"
            f"{c_file_output_block}"
            f"{h_file_output_block}"
            f"{identifier_block}"
            f"{testonly_block}"
            f"{flatten_block}"
            f"  PUBLIC\n)\n\n"
        )

    def iree_bitcode_library(self, name, arch, srcs, internal_hdrs=None, copts=None):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        arch_block = self._convert_string_arg_block("ARCH", arch, quote=False)
        hdrs_block = self._convert_srcs_block(internal_hdrs, block_name="INTERNAL_HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)

        self._converter.body += (
            f"iree_bitcode_library(\n"
            f"{name_block}"
            f"{arch_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f")\n\n"
        )

    def iree_cuda_bitcode_library(
        self, name, cuda_arch, srcs, internal_hdrs=None, copts=None
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        cuda_arch_block = self._convert_string_arg_block(
            "CUDA_ARCH", cuda_arch, quote=False
        )
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)

        self._converter.body += (
            f"iree_bitcode_library(\n"
            f"{name_block}"
            f"{cuda_arch_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f")\n\n"
        )

    def iree_link_bitcode(self, name, bitcode_files):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        bitcode_files_block = self._convert_srcs_block(
            [f.replace(":", "/") for f in bitcode_files]
        )

        self._converter.body += (
            f"iree_link_bitcode(\n" f"{name_block}" f"{bitcode_files_block}" f"\n)\n\n"
        )

    def iree_bytecode_module(
        self,
        name,
        src,
        module_name=None,
        flags=None,
        compile_tool=None,
        c_identifier=None,
        static_lib_path=None,
        deps=None,
        testonly=None,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        src_block = self._convert_string_arg_block("SRC", src)
        module_name_block = self._convert_string_arg_block(
            "MODULE_FILE_NAME", module_name
        )
        c_identifier_block = self._convert_string_arg_block(
            "C_IDENTIFIER", c_identifier
        )
        static_lib_block = self._convert_string_arg_block(
            "STATIC_LIB_PATH", static_lib_path
        )
        compile_tool_block = self._convert_target_block("COMPILE_TOOL", compile_tool)
        flags_block = self._convert_string_list_block("FLAGS", flags)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._converter.body += (
            f"iree_bytecode_module(\n"
            f"{name_block}"
            f"{src_block}"
            f"{module_name_block}"
            f"{c_identifier_block}"
            f"{compile_tool_block}"
            f"{static_lib_block}"
            f"{flags_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )

    def iree_flatbuffer_c_library(self, name, srcs, flatcc_args=None):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        flatcc_args_block = self._convert_string_list_block("FLATCC_ARGS", flatcc_args)

        self._converter.body += (
            f"flatbuffer_c_library(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{flatcc_args_block}"
            f"  PUBLIC\n)\n\n"
        )

    def gentbl_cc_library(
        self,
        name,
        tblgen,
        td_file,
        tbl_outs,
        td_srcs=None,
        deps=None,
        includes=None,
        strip_include_prefix=None,
        test=None,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        tblgen_block = self._convert_tblgen_block(tblgen)
        td_file_block = self._convert_td_file_block(td_file)
        outs_block = self._convert_tbl_outs_block(tbl_outs)

        self._converter.body += (
            f"iree_tablegen_library(\n"
            f"{name_block}"
            f"{td_file_block}"
            f"{outs_block}"
            f"{tblgen_block}"
            f")\n\n"
        )

    def iree_gentbl_cc_library(self, **kwargs):
        if self._should_skip_target(**kwargs):
            return
        # The bazel version of this rule adds some include directories and defs
        # that are implicitly handled by the cmake version.
        self.gentbl_cc_library(**kwargs)

    def iree_tablegen_doc(
        self,
        name,
        tblgen,
        td_file,
        tbl_outs,
        td_srcs=None,
        includes=None,
        deps=None,
        test=None,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        tblgen_block = self._convert_tblgen_block(tblgen)
        td_file_block = self._convert_td_file_block(td_file)
        outs_block = self._convert_tbl_outs_block(tbl_outs)

        self._converter.body += (
            f"iree_tablegen_doc(\n"
            f"{name_block}"
            f"{td_file_block}"
            f"{outs_block}"
            f"{tblgen_block}"
            f")\n\n"
        )

    def iree_lit_test_suite(
        self, name, srcs, tools=None, data=None, tags=None, timeout=None, **kwargs
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        tools_block = self._convert_target_list_block("TOOLS", tools)
        data_block = self._convert_target_list_block("DATA", data)
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._converter.body += (
            f"iree_lit_test_suite(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{tools_block}"
            f"{data_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f")\n\n"
        )

    def iree_check_single_backend_test_suite(
        self,
        name,
        srcs,
        target_backend,
        driver=None,
        compiler_flags=None,
        input_type=None,
        target_backends_and_drivers=None,
        runner_args=None,
        tags=None,
        target_cpu_features=None,
        timeout=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        target_backend_block = self._convert_string_arg_block(
            "TARGET_BACKEND", target_backend
        )
        driver_block = self._convert_string_arg_block("DRIVER", driver)
        compiler_flags_block = self._convert_string_list_block(
            "COMPILER_FLAGS", compiler_flags
        )
        input_type_block = self._convert_string_arg_block("INPUT_TYPE", input_type)
        runner_args_block = self._convert_string_list_block("RUNNER_ARGS", runner_args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        target_cpu_features_block = self._convert_string_arg_block(
            "TARGET_CPU_FEATURES", target_cpu_features
        )
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._converter.body += (
            f"iree_check_single_backend_test_suite(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{target_backend_block}"
            f"{driver_block}"
            f"{compiler_flags_block}"
            f"{input_type_block}"
            f"{runner_args_block}"
            f"{labels_block}"
            f"{target_cpu_features_block}"
            f"{timeout_block}"
            f")\n\n"
        )

    def iree_check_test_suite(
        self,
        name,
        srcs,
        target_backends_and_drivers=None,
        compiler_flags=None,
        runner_args=None,
        tags=None,
        target_cpu_features_variants=None,
        timeout=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_backends = None
        drivers = None
        if target_backends_and_drivers is not None:
            target_backends = [it[0] for it in target_backends_and_drivers]
            drivers = [it[1] for it in target_backends_and_drivers]

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        target_backends_block = self._convert_string_list_block(
            "TARGET_BACKENDS", target_backends
        )
        drivers_block = self._convert_string_list_block("DRIVERS", drivers)
        compiler_flags_block = self._convert_string_list_block(
            "COMPILER_FLAGS", compiler_flags
        )
        runner_args_block = self._convert_string_list_block("RUNNER_ARGS", runner_args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        target_cpu_features_variants_block = self._convert_string_list_block(
            "TARGET_CPU_FEATURES_VARIANTS", target_cpu_features_variants
        )
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._converter.body += (
            f"iree_check_test_suite(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{target_backends_block}"
            f"{drivers_block}"
            f"{compiler_flags_block}"
            f"{runner_args_block}"
            f"{labels_block}"
            f"{target_cpu_features_variants_block}"
            f"{timeout_block}"
            f")\n\n"
        )

    def iree_generated_trace_runner_test(
        self,
        name,
        generator,
        generator_args=None,
        trace_runner=None,
        target_backends_and_drivers=None,
        compiler_flags=None,
        runner_args=None,
        tags=None,
        target_cpu_features_variants=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_backends = None
        drivers = None
        if target_backends_and_drivers is not None:
            target_backends = [it[0] for it in target_backends_and_drivers]
            drivers = [it[1] for it in target_backends_and_drivers]

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        # For now we assume that the generator target is a py_binary with a single
        # source .py file named like it.
        generator_py = f"{generator.split(':')[-1]}.py"
        generator_block = self._convert_string_arg_block(
            "GENERATOR", generator_py, quote=True
        )
        generator_args_block = self._convert_string_list_block(
            "GENERATOR_ARGS", generator_args
        )
        trace_runner_block = self._convert_target_block("TRACE_RUNNER", trace_runner)
        target_backends_block = self._convert_string_list_block(
            "TARGET_BACKENDS", target_backends
        )
        drivers_block = self._convert_string_list_block("DRIVERS", drivers)
        compiler_flags_block = self._convert_string_list_block(
            "COMPILER_FLAGS", compiler_flags
        )
        runner_args_block = self._convert_string_list_block("RUNNER_ARGS", runner_args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        target_cpu_features_variants_block = self._convert_string_list_block(
            "TARGET_CPU_FEATURES_VARIANTS", target_cpu_features_variants
        )

        self._converter.body += (
            f"iree_generated_trace_runner_test(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{generator_args_block}"
            f"{trace_runner_block}"
            f"{target_backends_block}"
            f"{drivers_block}"
            f"{compiler_flags_block}"
            f"{runner_args_block}"
            f"{labels_block}"
            f"{target_cpu_features_variants_block}"
            f")\n\n"
        )

    def native_test(self, name, src, args=None, data=None, tags=None, timeout=None):
        if self._should_skip_target(tags=tags):
            return
        if data is not None:
            self._convert_unimplemented_function("native_test", name + " has data")

        name_block = self._convert_string_arg_block("NAME", name)
        test_binary_block = self._convert_single_target_block("SRC", src)
        args_block = self._convert_string_list_block("ARGS", args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._converter.body += (
            f"iree_native_test(\n"
            f"{name_block}"
            f"{args_block}"
            f"{test_binary_block}"
            f"{labels_block}"
            f")\n\n"
        )

    def cc_binary_benchmark(
        self,
        name,
        srcs=None,
        data=None,
        deps=None,
        copts=None,
        defines=None,
        linkopts=None,
        tags=None,
        testonly=True,
        # unused
        size="small",
        timeout=None,
    ):
        if self._should_skip_target(tags=tags):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        defines_block = self._convert_string_list_block("LINKOPTS", linkopts)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        labels_block = self._convert_string_list_block("LABELS", tags)

        self._converter.body += (
            f"iree_cc_binary_benchmark(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{defines_block}"
            f"{testonly_block}"
            f"{labels_block}"
            f")\n\n"
        )

    def iree_cmake_extra_content(self, content, inline=False):
        if inline:
            self._converter.body += f"\n{content}\n"
        else:
            self._converter.header += f"\n{content}\n"


class Converter(object):
    """Conversion state tracking and full file template substitution."""

    def __init__(self):
        # Header appears after the license block but before `iree_add_all_subdirs`.
        self.header = ""
        # Body appears after `iree_add_all_subdirs`.
        self.body = ""

        self.first_error = None

    def convert(self):
        converted_content = (
            f"{self.header}\n\n" f"iree_add_all_subdirs()\n\n" f"{self.body}"
        )

        # Cleanup newline characters. This is more convenient than ensuring all
        # conversions are careful with where they insert newlines.
        converted_content = converted_content.replace("\n\n\n", "\n")
        converted_content = converted_content.rstrip() + "\n"

        return converted_content


def GetDict(obj):
    ret = {}
    for k in dir(obj):
        if not k.startswith("_"):
            ret[k] = getattr(obj, k)
    return ret


def convert_build_file(
    build_file_code, repo_cfg, build_dir, allow_partial_conversion=False
):
    converter = Converter()
    # Allow overrides of TargetConverter and BuildFileFunctions from repo cfg.
    repo_map = getattr(repo_cfg, "REPO_MAP", {})
    target_converter = getattr(
        repo_cfg, "CustomTargetConverter", bazel_to_cmake_targets.TargetConverter
    )(repo_map=repo_map)
    build_file_functions = getattr(
        repo_cfg, "CustomBuildFileFunctions", BuildFileFunctions
    )(converter=converter, targets=target_converter, build_dir=build_dir)

    exec(build_file_code, GetDict(build_file_functions))
    converted_text = converter.convert()
    if not allow_partial_conversion and converter.first_error:
        raise converter.first_error  # pylint: disable=raising-bad-type
    return converted_text
