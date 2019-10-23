# Samples demonstrating use of the RT API.

load("///tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "simple_module_test_bytecode_module",
    srcs = ["simple_module_test.mlir"],
    cc_namespace = "iree::rt::samples",
)

cc_test(
    name = "bytecode_module_test",
    srcs = ["bytecode_module_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "///tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "///base:flatbuffer_util",
        "///base:status",
        "///base:status_matchers",
        "///hal:buffer_view",
        "///hal:driver_registry",
        "///rt",
        "///schemas",
        "///vm:bytecode_module",
        "///vm:sequencer_module",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "///hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        # TODO(benvanik): include SPIR-V.
        # "///hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ],
)

cc_test(
    name = "bytecode_module_api_test",
    srcs = ["bytecode_module_api_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "///tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "///base:api",
        "///hal:api",
        "///hal:driver_registry",
        "///rt:api",
        "///vm:api",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "///hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        # TODO(benvanik): include SPIR-V.
        # "///hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ],
)
