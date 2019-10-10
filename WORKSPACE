# Workspace file for the IREE project.
workspace(name = "iree")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(":repo_utils.bzl", "maybe")

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

###############################################################################
# io_bazel_rules_closure
# This is copied from https://github.com/tensorflow/tensorflow/blob/v2.0.0-alpha0/WORKSPACE.
# Dependency of:
#   TensorFlow (boilerplate for tf_workspace(), apparently)
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)
###############################################################################

###############################################################################
# Skylib
# Dependency of:
#   TensorFlow
http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()
###############################################################################

###############################################################################
# Bootstrap TensorFlow.
# Note that we ultimately would like to avoid doing this at the top level like
# this but need to unbundle some of the deps fromt the tensorflow repo first.
# In the mean-time: we're sorry.
# TODO(laurenzo): Come up with a way to make this optional. Also, see if we can
# get the TensorFlow tf_repositories() rule to use maybe() so we can provide
# local overrides safely.
maybe(local_repository,
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)

# Import all of the tensorflow dependencies.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")

tf_repositories()
###############################################################################

maybe(local_repository,
     name = "com_google_absl",
     path = "third_party/abseil-cpp",
)

maybe(local_repository,
     name = "com_google_googletest",
     path = "third_party/googletest",
)

# Note that TensorFlow provides this as "flatbuffers" which is wrong.
# It is only used for TFLite and may cause ODR issues if not fixed.
maybe(local_repository,
    name = "com_github_google_flatbuffers",
    path = "third_party/flatbuffers",
)

maybe(new_local_repository,
    name = "com_google_tracing_framework_cpp",
    path = "third_party/google_tracing_framework/bindings/cpp",
    build_file = "build_tools/third_party/google_tracing_framework_cpp/BUILD.overlay",
)

maybe(new_local_repository,
    name = "vulkan_headers",
    path = "third_party/vulkan_headers",
    build_file = "build_tools/third_party/vulkan_headers/BUILD.overlay",
)

maybe(new_local_repository,
    name = "vulkan_memory_allocator",
    path = "third_party/vulkan_memory_allocator",
    build_file = "build_tools/third_party/vulkan_memory_allocator/BUILD.overlay",
)
