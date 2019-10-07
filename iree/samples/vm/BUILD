# Samples demonstrating use of the VM API.

load("//iree/tools:compilation.bzl", "iree_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_module(
    name = "simple_module_test_module",
    srcs = ["simple_module_test.mlir"],
    cc_namespace = "iree::vm::samples",
)

cc_test(
    name = "simple_module_test",
    srcs = ["simple_module_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_module_cc",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings",
        "//iree/base:flatbuffer_util",
        "//iree/base:status_matchers",
        "//iree/hal:buffer_view",
        "//iree/hal:command_buffer",
        "//iree/hal:command_queue",
        "//iree/hal:driver_registry",
        "//iree/schemas",
        "//iree/base:status",
        "//iree/vm:fiber_state",
        "//iree/vm:instance",
        "//iree/vm:sequencer_context",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "//iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        # TODO(benvanik): include SPIR-V.
        # "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ],
)
