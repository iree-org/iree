# Main IREE build file.
# Note that project-wide, bazel repo aliases are used:
#   "@absl_py//absl"
#   "@com_google_absl//absl"
#   "@local_config_mlir//"
#   "@llvm//"
#   "@flatbuffers//:flatbuffers"
#   "@tensorflow//"
#
# Various scripts and helpers operate on these prefixes textually, so
# avoid doing any systematic construction that would break the matching.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Marker library which can be extended to provide flags for things that
# need to know the platform target.
cc_library(
    name = "target_config",
    defines = ["IREE_UNSPECIFIED_TARGET=1"],
)
