# Workspace file for the IREE project.
workspace(name = "iree")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Abseil depends on starlark rules that are currently maintained outside
# of Bazel.
# Source: https://github.com/abseil/abseil-cpp/blob/master/WORKSPACE
http_archive(
    name = "rules_cc",
    sha256 = "67412176974bfce3f4cf8bdaff39784a72ed709fc58def599d1f68710b58d68b",
    strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.zip",
        "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.zip",
    ],
)

local_repository(
    name = "com_google_absl",
    path = "third_party/abseil-cpp",
)

local_repository(
    name = "com_google_googletest",
    path = "third_party/googletest",
)

local_repository(
    name = "com_github_google_flatbuffers",
    path = "third_party/flatbuffers",
)

new_local_repository(
    name = "com_google_tracing_framework_cpp",
    path = "third_party/google_tracing_framework/bindings/cpp",
    build_file = "build_tools/third_party/google_tracing_framework_cpp/BUILD.overlay",
)

new_local_repository(
    name = "vulkan_headers",
    path = "third_party/vulkan_headers",
    build_file = "build_tools/third_party/vulkan_headers/BUILD.overlay",
)

new_local_repository(
    name = "vulkan_memory_allocator",
    path = "third_party/vulkan_memory_allocator",
    build_file = "build_tools/third_party/vulkan_memory_allocator/BUILD.overlay",
)
