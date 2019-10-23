# Misc tools used to optimize, translate, and evaluate IREE.
# Most of these are not designed to run on-device.

load("//:build_defs.google.bzl", "PLATFORM_VULKAN_DEPS")

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
        "///compiler/Transforms",
        "///compiler/Transforms/Interpreter",
        "///compiler/Transforms/Sequencer",
        "///compiler/Translation/SPIRV",
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
        "///base:source_location",
        "///rt",
        "///vm:sequencer_module",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Support",
        "///base:init",
        "///base:status",
        "///compiler/Translation/Sequencer",
        "///compiler/Translation/Interpreter",
        "///compiler/Translation/SPIRV",
        "///hal:buffer_view_string_util",
        "///hal:driver_registry",
        "///schemas",
        "///rt/debug:debug_server_flags",
    ] + PLATFORM_VULKAN_DEPS + [
        "///hal/interpreter:interpreter_driver_module",
        # TODO(b/142004903): enable when Dawn HAL implementation is functional
        # "///hal/dawn:dawn_driver_module",
        "///hal/vulkan:vulkan_driver_module",
    ],
)

cc_binary(
    name = "iree-translate",
    srcs = ["iree_translate_main.cc"],
    deps = [
        "///compiler/Translation/Interpreter",
        "///compiler/Translation/SPIRV",
        "///compiler/Translation/Sequencer",
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
        "///base:file_io",
        "///base:file_path",
        "///base:init",
        "///base:source_location",
        "///base:status",
        "///hal:buffer_view_string_util",
        "///hal:driver_registry",
        "///hal/interpreter:interpreter_driver_module",
        "///rt",
        "///rt/debug:debug_server_flags",
        "///schemas",
        "///vm:sequencer_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "benchmark_module",
    testonly = 1,
    srcs = ["benchmark_module.cc"],
    deps = [
        "///base:file_io",
        "///base:file_path",
        "///base:init",
        "///base:source_location",
        "///base:status",
        "///hal:buffer_view_string_util",
        "///hal:driver_registry",
        "///hal/interpreter:interpreter_driver_module",
        "///rt",
        "///rt/debug:debug_server_flags",
        "///schemas",
        "///vm:sequencer_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
    ],
)
