package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(benvanik): re-enable debugger after refactoring.
# cc_library(
#     name = "debug_client",
#     srcs = ["debug_client.cc"],
#     hdrs = ["debug_client.h"],
#     deps = [
#         ":debug_client_interface",
#         ":debug_client_tcp",  # build-cleaner: keep
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "@com_google_absl//absl/types:span",
#         "//iree/base:source_location",
#         "//iree/base:status",
#         "//iree/schemas",
#     ],
# )
#
# cc_library(
#     name = "debug_client_interface",
#     hdrs = ["debug_client.h"],
#     deps = [
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "@com_google_absl//absl/types:span",
#         "//iree/base:status",
#         "//iree/schemas",
#     ],
# )
#
# cc_library(
#     name = "debug_client_tcp",
#     srcs = ["debug_client_tcp.cc"],
#     deps = [
#         ":debug_client_interface",
#         ":debug_tcp_util",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:span",
#         "@com_github_google_flatbuffers//:flatbuffers",
#         "//iree/base:flatbuffer_util",
#         "//iree/base:status",
#         "//iree/rt",
#         "//iree/schemas",
#     ],
# )
#
# cc_library(
#     name = "debug_server",
#     hdrs = ["debug_server.h"],
#     deps = [
#         ":debug_server_interface",
#         "//third_party/flatbuffers:flatbuffers",
#         "//iree/schemas",
#         "//iree/base:status",
#     ] + select({
#         "//iree:debug": [":debug_server_tcp"],
#         "//conditions:default": [":debug_server_disabled"],
#     }),
# )

cc_library(
    name = "debug_server",
    hdrs = ["debug_server.h"],
    deps = [
        ":debug_server_disabled",
        ":debug_server_interface",
        "//iree/base:status",
    ],
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

# TODO(benvanik): re-enable debugger after refactoring.
# cc_library(
#     name = "debug_server_tcp",
#     srcs = ["debug_server_tcp.cc"],
#     deps = [
#         ":debug_server_interface",
#         ":debug_service",
#         ":debug_tcp_util",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/synchronization",
#         "@com_github_google_flatbuffers//:flatbuffers",
#         "//iree/base:status",
#         "//iree/schemas",
#     ],
# )

cc_library(
    name = "debug_server_flags",
    srcs = ["debug_server_flags.cc"],
    hdrs = ["debug_server_flags.h"],
    deps = [
        ":debug_server",
        "//iree/base:memory",
        "//iree/base:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(benvanik): re-enable debugger after refactoring.
# cc_library(
#     name = "debug_server_flags",
#     srcs = ["debug_server_flags.cc"],
#     hdrs = ["debug_server_flags.h"],
#     copts = select({
#         "//iree:debug": [
#             "-DIREE_DEBUG_EMBEDDED_APP_PRESENT=1",
#         ],
#         "//conditions:default": [],
#     }),
#     deps = [
#         ":debug_server",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/strings",
#         "//iree/base:memory",
#         "//iree/base:status",
#     ] + select({
#         "//iree:debug": [
#             "//iree/tools/debugger:debug_app_embedded",
#             "//third_party/GL/native:EGL",  # build-cleaner: keep
#             "//third_party/GL/native:GLESv2",  # build-cleaner: keep
#         ],
#         "//conditions:default": [],
#     }),
# )
#
# cc_library(
#     name = "debug_service",
#     srcs = ["debug_service.cc"],
#     hdrs = ["debug_service.h"],
#     deps = [
#         ":debug_session",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/synchronization",
#         "@com_github_google_flatbuffers//:flatbuffers",
#         "//iree/base:flatbuffer_util",
#         "//iree/base:source_location",
#         "//iree/base:status",
#         "//iree/rt",
#         "//iree/schemas",
#         "//iree/schemas:reflection_data",
#     ],
# )
#
# cc_library(
#     name = "debug_session",
#     srcs = ["debug_session.cc"],
#     hdrs = ["debug_session.h"],
#     deps = [
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/synchronization",
#         "//iree/base:source_location",
#         "//iree/base:status",
#         "//iree/rt",
#         "//iree/schemas",
#     ],
# )
#
# cc_library(
#     name = "debug_tcp_util",
#     hdrs = ["debug_tcp_util.h"],
#     deps = [
#         "@com_github_google_flatbuffers//:flatbuffers",
#         "//iree/base:status",
#         "//iree/schemas",
#     ],
# )
