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

load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "add_dynamic",
    testonly = True,
    src = "add_dynamic.mlir",
    c_identifier = "iree_tflite_testdata_add_dynamic",
    flags = [
        "-iree-native-bindings-support=false",
        "-iree-tflite-bindings-support",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmvx",
    ],
)

iree_bytecode_module(
    name = "add_multi",
    testonly = True,
    src = "add_multi.mlir",
    c_identifier = "iree_tflite_testdata_add_multi",
    flags = [
        "-iree-native-bindings-support=false",
        "-iree-tflite-bindings-support",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmvx",
    ],
)

iree_bytecode_module(
    name = "add_static",
    testonly = True,
    src = "add_static.mlir",
    c_identifier = "iree_tflite_testdata_add_static",
    flags = [
        "-iree-native-bindings-support=false",
        "-iree-tflite-bindings-support",
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmvx",
    ],
)
