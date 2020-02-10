This folder contains experimental subprojects related to IREE and MLIR. These
are not yet stable and supported and may not always be working. We may keep the
build bots green for certain configurations but would prefer not to take on too
much maintence overhead for things unless they are on a path to leaving
experimental. Please use forks of the repository for purely
experimental/personal work.

**NOTE**: not all projects require a cmake build. If you are adding a directory
that is only expected to build with bazel (such as something depending on
TensorFlow) you can ignore cmake.
