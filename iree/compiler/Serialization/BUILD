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
        "VMDeviceTableBuilder.cpp",
        "VMExecutableTableBuilder.cpp",
        "VMFunctionBuilder.cpp",
        "VMFunctionTableBuilder.cpp",
        "VMModuleBuilder.cpp",
        "VMSourceMapBuilder.cpp",
    ],
    hdrs = [
        "BytecodeTables.h",
        "BytecodeWriter.h",
        "VMDeviceTableBuilder.h",
        "VMExecutableTableBuilder.h",
        "VMFunctionBuilder.h",
        "VMFunctionTableBuilder.h",
        "VMModuleBuilder.h",
        "VMSourceMapBuilder.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Utils",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "//iree/schemas/bytecode:sequencer_bytecode_v0",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
)
