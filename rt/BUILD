# Runtime API for interacting with IREE modules and invoking functions.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "rt",
    srcs = [
        "context.cc",
        "function.cc",
        "instance.cc",
        "invocation.cc",
        "source_location.cc",
        "stack_frame.cc",
        "stack_trace.cc",
    ],
    hdrs = [
        "context.h",
        "function.h",
        "function_signature.h",
        "instance.h",
        "invocation.h",
        "module.h",
        "module_signature.h",
        "policy.h",
        "source_location.h",
        "source_resolver.h",
        "stack_frame.h",
        "stack_trace.h",
    ],
    deps = [
        "//iree/base:intrusive_list",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:buffer_view",
        "//iree/hal:device_manager",
        "//iree/rt/debug:debug_server_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)
