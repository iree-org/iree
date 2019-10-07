# Main IREE build file.

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
