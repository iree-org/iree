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
        "//iree/compiler/Transforms",
        "//iree/compiler/Transforms/Interpreter",
        "//iree/compiler/Transforms/Sequencer",
        "//iree/compiler/Translation/SPIRV",
        "@llvm//:support",
        "@local_config_mlir//:AffineDialectRegistration",
        "@local_config_mlir//:MlirOptLib",
        "@local_config_mlir//:MlirOptMain",
        "@local_config_mlir//:StandardDialectRegistration",
        "@tensorflow///compiler/mlir/xla:hlo",
        "@tensorflow///compiler/mlir/xla:xla_dialect_registration",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["run_mlir_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:source_location",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Support",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/compiler/Translation/Sequencer",
        "//iree/compiler/Translation/Interpreter",
        "//iree/compiler/Translation/SPIRV",
        "//iree/hal:buffer_view_string_util",
        "//iree/hal:driver_registry",
        "//iree/hal/interpreter:interpreter_driver_module",
        "//iree/hal/vulkan:vulkan_driver_module",
        # TODO(b/142004903): enable when Dawn HAL implementation is functional
        # "//iree/hal/dawn:dawn_driver_module",
        "//iree/schemas",
        "//iree/vm:bytecode_tables_sequencer",
        "//iree/vm:fiber_state",
        "//iree/vm:instance",
        "//iree/vm:module",
        "//iree/vm:module_printer",
        "//iree/vm:sequencer_context",
        "//iree/vm/debug:debug_server_flags",
    ] + PLATFORM_VULKAN_DEPS,
)

cc_binary(
    name = "iree-translate",
    srcs = ["iree_translate_main.cc"],
    deps = [
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
        "@tensorflow///compiler/mlir/xla:xla_dialect_registration",
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
        "//iree/schemas",
        "//iree/vm:bytecode_printer",
        "//iree/vm:bytecode_tables_sequencer",
        "//iree/vm:fiber_state",
        "//iree/vm:function",
        "//iree/vm:instance",
        "//iree/vm:module",
        "//iree/vm:module_printer",
        "//iree/vm:sequencer_context",
        "//iree/vm/debug:debug_server_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)
