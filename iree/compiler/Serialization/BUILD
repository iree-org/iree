# Copyright 2019 Google LLC
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

# Serialization for the VM bytecode.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Serialization",
    srcs = [
        "BytecodeTables.cpp",
        "BytecodeWriter.cpp",
        "VMFunctionBuilder.cpp",
        "VMFunctionTableBuilder.cpp",
        "VMModuleBuilder.cpp",
    ],
    hdrs = [
        "BytecodeTables.h",
        "BytecodeWriter.h",
        "VMFunctionBuilder.h",
        "VMFunctionTableBuilder.h",
        "VMModuleBuilder.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Utils",
        "//iree/schemas:interpreter_module_def_cc_fbs",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)
