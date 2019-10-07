# Misc tools used to optimize, translate, and evaluate IREE.
# Most of these are not designed to run on-device.

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
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:AffineDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:MlirOptLib",
        "//third_party/llvm/llvm/projects/google_mlir:MlirOptMain",
        "//third_party/llvm/llvm/projects/google_mlir:StandardDialectRegistration",
        "//third_party/tensorflow/compiler/mlir/xla:hlo",
        "//third_party/tensorflow/compiler/mlir/xla:xla_dialect_registration",
    ],
)

cc_binary(
    name = "iree-run-mlir",
    srcs = ["run_mlir_main.cc"],
    deps = [
        "//third_party/absl/flags:flag",
        "//third_party/absl/strings",
        "//iree/base:source_location",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:Parser",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/compiler/Translation/Sequencer",
        "//iree/compiler/Translation/Interpreter",
        "//iree/compiler/Translation/SPIRV",
        "//iree/hal:buffer_view_string_util",
        "//iree/hal:driver_registry",
        "//iree/hal/interpreter:interpreter_driver_module",
        # TODO(b/138220713): enable when we have swiftshader support.
        # "//iree/hal/vulkan:vulkan_driver_module",
        "//iree/schemas",
        "//iree/vm:bytecode_tables_sequencer",
        "//iree/vm:fiber_state",
        "//iree/vm:instance",
        "//iree/vm:module",
        "//iree/vm:module_printer",
        "//iree/vm:sequencer_context",
        "//iree/vm/debug:debug_server_flags",
    ],
)

cc_binary(
    name = "iree-translate",
    deps = [
        "//iree/compiler/Translation/Interpreter",
        "//iree/compiler/Translation/SPIRV",
        "//iree/compiler/Translation/Sequencer",
        "//third_party/llvm/llvm/projects/google_mlir:AffineDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:StandardDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:tools/mlir-translate/mlir-translate",
        "//third_party/tensorflow/compiler/mlir/xla:xla_dialect_registration",
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
        "//third_party/absl/flags:flag",
        "//third_party/absl/strings",
    ],
)
