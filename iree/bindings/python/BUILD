package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

DRIVER_DEPS = [
    "//iree/hal/interpreter:interpreter_driver_module",
    "//iree/hal/vulkan:vulkan_driver_module",
]

py_extension(
    name = "binding",
    srcs = [
        "binding.cc",
        "binding.h",
        "compiler.cc",
        "compiler.h",
        "status_utils.cc",
        "status_utils.h",
    ],
    copts = [
        "-fexceptions",
        #"-fno-strict-aliasing",
        #"-Wno-c++98-c++11-compat",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/compiler/Translation/Sequencer",
        "//iree/schemas",
        "//third_party/pybind11",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    deps = [
        ":binding",
        "//third_party/py/absl/testing:absltest",
    ],
)
