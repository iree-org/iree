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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# TFLite C API compatibility shim
#===------------------------------------------------------------------------===#
# See README.md for more information.
# Use this *in place of tflite*; do not link both this and tflite into your
# binary (you'll get ODR violations at the least, if not worse).

cc_library(
    name = "shim",
    srcs = [
        "interpreter.c",
        "interpreter.h",
        "model.c",
        "model.h",
        "options.c",
        "options.h",
        "shim.c",
        "shim.h",
        "tensor.c",
        "tensor.h",
    ],
    hdrs = [
        "include/tensorflow/lite/c/c_api.h",
        "include/tensorflow/lite/c/c_api_experimental.h",
        "include/tensorflow/lite/c/common.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:internal",
        "//iree/base:threading",
        "//iree/base:tracing",
        "//iree/hal:api",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

#===------------------------------------------------------------------------===#
# Tests for our own unique functionality and tflite compatibility
#===------------------------------------------------------------------------===#

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    # TODO(benvanik): bazel/cmake to include this generated file.
    # Checked in right now until I can figure it out.
    # data = [
    #     "//bindings/tflite/testdata:add.vmfb",
    # ],
    deps = [
        ":shim",
        "//bindings/tflite/testdata:add_dynamic_cc",
        "//bindings/tflite/testdata:add_multi_cc",
        "//bindings/tflite/testdata:add_static_cc",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
