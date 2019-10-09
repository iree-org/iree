# Samples demonstrating use of the HAL API.
# These do not rely on higher layers of the system (such as the VM or runtime).

load("//iree:build_defs.bzl", "PLATFORM_VULKAN_TEST_DEPS")
load("//iree/tools:compilation.bzl", "iree_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_module(
    name = "simple_compute_test_module",
    srcs = ["simple_compute_test.mlir"],
    cc_namespace = "iree::hal::samples",
)

cc_test(
    name = "simple_compute_test",
    srcs = ["simple_compute_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_compute_test_module_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//iree/base:flatbuffer_util",
        "//iree/base:status_matchers",
        "//iree/hal:command_buffer",
        "//iree/hal:command_queue",
        "//iree/hal:driver_registry",
        "//iree/schemas",
        "//iree/base:status",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "//iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep

        # TODO(b/142004903): enable when Dawn HAL implementation is functional
        # "//iree/hal/dawn:dawn_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)
