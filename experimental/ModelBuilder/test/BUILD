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

# Tests for end-to-end IREE support starting from the XLA HLO dialect.

load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = glob(["*.cpp"]),
    data = [
        ":runtime-support.so",
        ":test-mnist-jit",
        ":test-simple-jit",
        "//iree/tools:IreeFileCheck",
    ],
    tags = [
        "noga",
    ],
)

cc_binary(
    name = "test-mnist-jit",
    srcs = ["TestMNISTJIT.cpp"],
    tags = [
        "noga",
    ],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopOpsTransforms",
        "@llvm-project//mlir:mlir_runner_utils",
        "@llvm-project//mlir:mlir_runner_utils_header",
    ],
)

cc_binary(
    name = "test-simple-jit",
    srcs = ["TestSimpleJIT.cpp"],
    tags = [
        "noga",
    ],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopOpsTransforms",
    ],
)

cc_binary(
    name = "runtime-support.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)
