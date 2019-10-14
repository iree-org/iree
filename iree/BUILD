# Main IREE build file.
# Note that project-wide, bazel repo aliases are used:
#   "@absl_py//absl"
#   "@com_google_absl//absl"
#   "@local_config_mlir//"
#   "@llvm//"
#   "@com_github_google_flatbuffers//:flatbuffers"
#   "@org_tensorflow//tensorflow"
#
# Various scripts and helpers operate on these prefixes textually, so
# avoid doing any systematic construction that would break the matching.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Enables the debug service and other profiling features.
# $ bazel build --define=IREE_DEBUG=1 :some_target
config_setting(
    name = "debug",
    define_values = {"IREE_DEBUG": "1"},
)

# Marker library which can be extended to provide flags for things that
# need to know the platform target.
cc_library(
    name = "target_config",
    defines = ["IREE_UNSPECIFIED_TARGET=1"],
)
