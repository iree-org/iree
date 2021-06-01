load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
# absl and benchmark have thread dependency.
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()
""",
    inline = True,
)

cc_library(
    name = "strings_module",
    srcs = [
        "api.cc",
        "strings_module.cc",
    ],
    hdrs = [
        "api.h",
        "api_detail.h",
        "strings_module.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:cc",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_cmake_extra_content(
    content = """
if (NOT ${IREE_BUILD_COMPILER})
  return()
endif()
""",
    inline = True,
)

cc_test(
    name = "strings_module_test",
    srcs = ["strings_module_test.cc"],
    deps = [
        ":strings_module",
        ":strings_module_test_module_c",
        "//iree/base",
        "//iree/base:logging",
        "//iree/hal",
        "//iree/hal/vmvx/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_bytecode_module(
    name = "strings_module_test_module",
    testonly = True,
    src = "strings_module_test.mlir",
    c_identifier = "iree_strings_module_test_module",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
    ],
)
