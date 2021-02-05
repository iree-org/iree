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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if (NOT ${IREE_BUILD_COMPILER} OR NOT ${IREE_BUILD_TESTS})
  return()
endif()
""",
    inline = True,
)

cc_embed_data(
    name = "all_bytecode_modules_cc",
    srcs = [
        ":arithmetic_ops.vmfb",
        ":arithmetic_ops_i64.vmfb",
        ":comparison_ops.vmfb",
        ":control_flow_ops.vmfb",
        ":list_ops.vmfb",
        ":shift_ops.vmfb",
    ],
    cc_file_output = "all_bytecode_modules.cc",
    cpp_namespace = "iree::vm::test",
    flatten = True,
    h_file_output = "all_bytecode_modules.h",
)

iree_bytecode_module(
    name = "arithmetic_ops",
    src = "arithmetic_ops.mlir",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_bytecode_module(
    name = "arithmetic_ops_i64",
    src = "arithmetic_ops_i64.mlir",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_bytecode_module(
    name = "comparison_ops",
    src = "comparison_ops.mlir",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_bytecode_module(
    name = "control_flow_ops",
    src = "control_flow_ops.mlir",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_bytecode_module(
    name = "list_ops",
    src = "list_ops.mlir",
    cc_namespace = "iree::vm::test",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_bytecode_module(
    name = "shift_ops",
    src = "shift_ops.mlir",
    cc_namespace = "iree::vm::test",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)
