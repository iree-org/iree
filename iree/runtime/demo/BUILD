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

load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Hello World!
#===------------------------------------------------------------------------===#

cc_binary(
    name = "hello_world_embedded",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data directly from memory.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA",
    ],
    deps = [
        "//iree/runtime",
        "//iree/runtime/testdata:simple_mul_module_c",
    ],
)

run_binary_test(
    name = "hello_world_embedded_test",
    test_binary = ":hello_world_embedded",
)

cc_binary(
    name = "hello_world_file",
    srcs = ["hello_world_explained.c"],
    defines = [
        # Load data from a file passed on the command line.
        "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG",
    ],
    deps = [
        "//iree/runtime",
    ],
)

# TODO(benvanik): run_binary_test that passes the file as a flag. Right now we
# can't specify data through run_binary_test, though, so this isn't possible to
# automate.

cc_binary(
    name = "hello_world_terse",
    srcs = ["hello_world_terse.c"],
    deps = [
        "//iree/runtime",
        "//iree/runtime/testdata:simple_mul_module_c",
    ],
)

run_binary_test(
    name = "hello_world_terse_test",
    test_binary = ":hello_world_terse",
)
