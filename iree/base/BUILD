# Common types and utilities used in the IREE codebase.

load("///iree:build_defs.bzl", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":api_hdrs",
        ":api_util",
        ":file_mapping",
        ":tracing",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
)

cc_library(
    name = "api_util",
    hdrs = ["api_util.h"],
    deps = [
        ":api_hdrs",
        ":logging",
        ":shape",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "arena",
    srcs = ["arena.cc"],
    hdrs = ["arena.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "arena_test",
    srcs = ["arena_test.cc"],
    deps = [
        ":arena",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bitfield",
    hdrs = ["bitfield.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "bitfield_test",
    srcs = ["bitfield_test.cc"],
    deps = [
        ":bitfield",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "file_io",
    hdrs = ["file_io.h"],
    deps = [
        ":status",
        ":target_platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + platform_trampoline_deps("file_io"),
)

cc_library(
    name = "file_io_hdrs",
    hdrs = ["file_io.h"],
    deps = [":status"],
)

cc_library(
    name = "file_mapping",
    hdrs = ["file_mapping.h"],
    deps = [
        ":ref_ptr",
        ":status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + platform_trampoline_deps("file_mapping"),
)

cc_library(
    name = "file_mapping_hdrs",
    hdrs = ["file_mapping.h"],
    deps = [
        ":ref_ptr",
        ":status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_path",
    srcs = ["file_path.cc"],
    hdrs = ["file_path.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "flatbuffer_util",
    srcs = ["flatbuffer_util.cc"],
    hdrs = ["flatbuffer_util.h"],
    deps = [
        ":file_mapping",
        ":memory",
        ":source_location",
        ":status",
        ":tracing",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "init",
    hdrs = ["init.h"],
    deps = platform_trampoline_deps("init"),
)

cc_library(
    name = "intrusive_list",
    hdrs = [
        "intrusive_list.h",
        "intrusive_list_ref_ptr.inc",
        "intrusive_list_unique_ptr.inc",
    ],
    deps = [
        ":logging",
        ":ref_ptr",
    ],
)

cc_test(
    name = "intrusive_list_test",
    srcs = [
        "intrusive_list_ref_ptr_test.cc",
        "intrusive_list_test.cc",
        "intrusive_list_unique_ptr_test.cc",
    ],
    deps = [
        ":intrusive_list",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = platform_trampoline_deps("logging"),
)

cc_library(
    name = "math",
    hdrs = ["math.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "memory",
    hdrs = ["memory.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ref_ptr",
    hdrs = ["ref_ptr.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "ref_ptr_test",
    size = "small",
    srcs = ["ref_ptr_test.cc"],
    deps = [
        ":ref_ptr",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        ":logging",
        ":source_location",
        ":status",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":shape",
        ":status",
        ":status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = platform_trampoline_deps("source_location"),
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [
        ":source_location",
    ] + platform_trampoline_deps("status"),
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    hdrs = ["status_matchers.h"],
    deps = platform_trampoline_deps("status_matchers"),
)

cc_library(
    name = "target_platform",
    hdrs = ["target_platform.h"],
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "tracing",
    hdrs = ["tracing.h"],
    deps = [
        "//:target_config",
        "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
    ] + select({
        "@com_google_tracing_framework_cpp//:wtf_enable": [":tracing_enabled"],
        "//conditions:default": [":tracing_disabled"],
    }),
)

cc_library(
    name = "tracing_disabled",
    srcs = [
        "tracing.h",
        "tracing_disabled.cc",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":init",
        ":logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tracing_enabled",
    srcs = [
        "tracing.cc",
        "tracing.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":file_io",
        ":file_path",
        ":init",
        ":logging",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_tracing_framework_cpp//:tracing_framework_bindings_cpp",
    ],
    alwayslink = 1,
)

# Dependent code has been removed and wait_handle is currently incompatible
# with Windows, so excluding entirely.
# See google/iree/65
# cc_library(
#     name = "wait_handle",
#     srcs = ["wait_handle.cc"],
#     hdrs = ["wait_handle.h"],
#     deps = [
#         ":logging",
#         ":ref_ptr",
#         ":source_location",
#         ":status",
#         ":time",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/container:fixed_array",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#     ],
# )

# cc_test(
#     name = "wait_handle_test",
#     srcs = ["wait_handle_test.cc"],
#     deps = [
#         ":status",
#         ":status_matchers",
#         ":wait_handle",
#         "@com_google_absl//absl/time",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
