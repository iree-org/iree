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

# Misc tools used to optimize, translate, and evaluate IREE.
# Most of these are not designed to run on-device.

load("//iree:build_defs.bzl", "PLATFORM_VULKAN_DEPS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "run_lit.sh",
    "sanitizer_suppressions.txt",
])

cc_binary(
    name = "iree-opt",
    deps = [
        "//integrations/tensorflow/compiler:tensorflow",
        "//iree/compiler/Dialect",
        "//iree/compiler/Dialect/Flow/Analysis",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Transforms",
        "//iree/compiler/Transforms/Interpreter",
        "//iree/compiler/Transforms/Sequencer",
        "//iree/compiler/Translation/SPIRV",
        "@llvm//:support",
        "@local_config_mlir//:AffineDialectRegistration",
        "@local_config_mlir//:MlirOptLib",
        "@local_config_mlir//:MlirOptMain",
        "@local_config_mlir//:StandardDialectRegistration",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_dialect_registration",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["run_mlir_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:source_location",
        "//iree/rt",
        "//iree/vm:sequencer_module",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Support",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/compiler/Translation/Sequencer",
        "//iree/compiler/Translation/Interpreter",
        "//iree/compiler/Translation/SPIRV",
        "//iree/compiler/Translation:IREEVM",
        "//iree/hal:buffer_view_string_util",
        "//iree/hal:driver_registry",
        "//iree/schemas",
        "//iree/rt/debug:debug_server_flags",
    ] + PLATFORM_VULKAN_DEPS + [
        "//iree/hal/interpreter:interpreter_driver_module",
        # TODO(b/142004903): enable when Dawn HAL implementation is functional
        # "//iree/hal/dawn:dawn_driver_module",
        "//iree/hal/vulkan:vulkan_driver_module",
    ],
)

cc_binary(
    name = "iree-tblgen",
    linkopts = [
        "-lpthread",
    ],
    deps = [
        "//iree/compiler/Dialect/VM/Tools",
        "@local_config_mlir//:MlirTableGenMain",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TableGen",
    ],
)

cc_binary(
    name = "iree-translate",
    srcs = ["iree_translate_main.cc"],
    deps = [
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Translation:IREEVM",
        "//iree/compiler/Translation/Interpreter",
        "//iree/compiler/Translation/SPIRV",
        "//iree/compiler/Translation/Sequencer",
        "@llvm//:support",
        "@local_config_mlir//:AffineDialectRegistration",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TranslateClParser",
        "@local_config_mlir//:Translation",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_dialect_registration",
    ],
)

cc_binary(
    name = "run_module",
    srcs = ["run_module_main.cc"],
    deps = [
        "//iree/base:file_io",
        "//iree/base:file_path",
        "//iree/base:init",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/hal:buffer_view_string_util",
        "//iree/hal:driver_registry",
        "//iree/hal/interpreter:interpreter_driver_module",
        "//iree/rt",
        "//iree/rt/debug:debug_server_flags",
        "//iree/schemas",
        "//iree/vm:sequencer_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "iree_dump_module",
    srcs = ["dump_module.cc"],
    deps = [
        "//iree/base:file_io",  # build-cleaner: keep
        "//iree/base:flatbuffer_util",
        "//iree/base:init",
        "//iree/schemas",
        "//iree/schemas:reflection_data",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)
