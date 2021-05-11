# Copyright 2021 Google LLC
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

iree_bytecode_module(
    name = "simple_mul_module",
    src = "simple_mul.mlir",
    c_identifier = "iree_runtime_testdata_simple_mul_module",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmla",
    ],
)
