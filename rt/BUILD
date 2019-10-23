# Runtime API for interacting with IREE modules and invoking functions.

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
        ":rt",
        "///base:api",
        "///base:api_util",
        "///base:tracing",
        "///hal:api",
        "///hal:buffer_view",
        "///hal:driver_registry",
        "///rt/debug:debug_server_interface",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
    deps = [
        "///base:api_hdrs",
        "///hal:api_hdrs",
    ],
)

cc_library(
    name = "rt",
    srcs = [
        "context.cc",
        "function.cc",
        "instance.cc",
        "invocation.cc",
        "module_printer.cc",
        "source_location.cc",
        "stack.cc",
        "stack_frame.cc",
        "stack_trace.cc",
    ],
    hdrs = [
        "context.h",
        "disassembler.h",
        "function.h",
        "function_signature.h",
        "instance.h",
        "invocation.h",
        "module.h",
        "module_printer.h",
        "module_signature.h",
        "policy.h",
        "source_location.h",
        "source_resolver.h",
        "stack.h",
        "stack_frame.h",
        "stack_trace.h",
    ],
    deps = [
        "///base:bitfield",
        "///base:intrusive_list",
        "///base:ref_ptr",
        "///base:status",
        "///base:tracing",
        "///hal:buffer_view",
        "///hal:device_manager",
        "///rt/debug:debug_server_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)
