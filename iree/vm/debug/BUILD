package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "debug_client",
    srcs = ["debug_client.cc"],
    hdrs = ["debug_client.h"],
    deps = [
        ":debug_client_interface",
        ":debug_client_tcp",  # build-cleaner: keep
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/schemas",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "debug_client_interface",
    hdrs = ["debug_client.h"],
    deps = [
        "//iree/base:status",
        "//iree/schemas",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "debug_client_tcp",
    srcs = ["debug_client_tcp.cc"],
    deps = [
        ":debug_client_interface",
        ":debug_tcp_util",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/schemas",
        "//iree/vm:module",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "debug_server",
    hdrs = ["debug_server.h"],
    deps = [
        ":debug_server_interface",
        "@com_github_google_flatbuffers//:flatbuffers",
        "//iree/schemas",
        "//iree/base:status",
    ] + select({
        "//iree/vm:debug": [":debug_server_tcp"],
        "//conditions:default": [":debug_server_disabled"],
    }),
)

cc_library(
    name = "debug_server_interface",
    hdrs = ["debug_server.h"],
    deps = ["//iree/base:status"],
)

cc_library(
    name = "debug_server_disabled",
    srcs = ["debug_server_disabled.cc"],
    deps = [
        ":debug_server_interface",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "debug_server_tcp",
    srcs = ["debug_server_tcp.cc"],
    deps = [
        ":debug_server_interface",
        ":debug_service",
        ":debug_tcp_util",
        "//iree/base:status",
        "//iree/schemas",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "debug_server_flags",
    srcs = ["debug_server_flags.cc"],
    hdrs = ["debug_server_flags.h"],
    copts = select({
        "//iree/vm:debug": [
            "-DIREE_DEBUG_EMBEDDED_APP_PRESENT=1",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":debug_server",
        "//iree/base:memory",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:status",
    ] + select({
        "//iree/vm:debug": [
            "//iree/tools/debugger:debug_app_embedded",
            "//third_party/GL/native:EGL",  # build-cleaner: keep
            "//third_party/GL/native:GLESv2",  # build-cleaner: keep
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "debug_service",
    srcs = ["debug_service.cc"],
    hdrs = ["debug_service.h"],
    deps = [
        ":debug_session",
        "//iree/base:flatbuffer_util",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/schemas",
        "//iree/schemas:reflection_data",
        "//iree/vm:fiber_state",
        "//iree/vm:instance",
        "//iree/vm:sequencer_context",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "debug_session",
    srcs = ["debug_session.cc"],
    hdrs = ["debug_session.h"],
    deps = [
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/schemas",
        "//iree/vm:fiber_state",
        "//iree/vm:sequencer_context",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "debug_tcp_util",
    hdrs = ["debug_tcp_util.h"],
    deps = [
        "//iree/base:status",
        "//iree/schemas",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)
