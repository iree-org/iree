# Samples demonstrating use of the RT API.

load("///iree/tools:compilation.bzl", "iree_bytecode_module")

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
        "///iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "///iree/base:flatbuffer_util",
        "///iree/base:status",
        "///iree/base:status_matchers",
        "///iree/hal:buffer_view",
        "///iree/hal:driver_registry",
        "///iree/rt",
        "///iree/schemas",
        "///iree/vm:bytecode_module",
        "///iree/vm:sequencer_module",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "///iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        # TODO(benvanik): include SPIR-V.
        # "///iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ],
)

cc_test(
    name = "bytecode_module_api_test",
    srcs = ["bytecode_module_api_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "///iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "///iree/base:api",
        "///iree/hal:api",
        "///iree/hal:driver_registry",
        "///iree/rt:api",
        "///iree/vm:api",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "///iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        # TODO(benvanik): include SPIR-V.
        # "///iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ],
)
