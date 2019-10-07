# Implementations for iree/base/

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "file_handle_win32",
    srcs = ["file_handle_win32.cc"],
    hdrs = ["file_handle_win32.h"],
    deps = [
        "//iree/base:status",
        "//iree/base:target_platform",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
    ],
)

cc_library(
    name = "file_io_internal",
    srcs = [
        "file_io_posix.cc",
        "file_io_win32.cc",
    ],
    deps = [
        ":file_handle_win32",
        "//iree/base:file_io_hdrs",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
    ],
)

cc_library(
    name = "file_mapping_internal",
    srcs = [
        "file_mapping_posix.cc",
        "file_mapping_win32.cc",
    ],
    deps = [
        ":file_handle_win32",
        "//iree/base:file_mapping_hdrs",
        "//iree/base:target_platform",
        "//iree/base:tracing",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
    ],
)

cc_library(
    name = "init_internal",
    srcs = ["init_internal.cc"],
    hdrs = ["init_internal.h"],
    deps = [
        "//iree/base:target_platform",
        "//third_party/absl/flags:parse",
    ],
)

cc_library(
    name = "logging_internal",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "//third_party/absl/base:core_headers",
        "//third_party/absl/flags:flag",
    ],
)

cc_library(
    name = "source_location_internal",
    hdrs = ["source_location.h"],
)

cc_library(
    name = "status_internal",
    srcs = [
        "status.cc",
        "status_builder.cc",
        "status_errno.cc",
        "status_errors.cc",
        "status_win32_errors.cc",
        "statusor.cc",
    ],
    hdrs = [
        "status.h",
        "status_builder.h",
        "status_errno.h",
        "status_errors.h",
        "status_macros.h",
        "status_win32_errors.h",
        "statusor.h",
    ],
    deps = [
        ":logging_internal",
        "//iree/base:source_location",
        "//iree/base:target_platform",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/debugging:stacktrace",
        "//third_party/absl/flags:flag",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
    ],
)

cc_library(
    name = "status_matchers_internal",
    testonly = 1,
    hdrs = ["status_matchers.h"],
    deps = [
        "//iree/base:status",
        "//testing/base/public:gunit_for_library",
        "//third_party/absl/strings",
        "//third_party/absl/types:optional",
    ],
)
