# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for configuring IREE and dependent project WORKSPACE files."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_skylib//lib:paths.bzl", "paths")

def configure_iree_submodule_deps(iree_repo_alias = "@", iree_path = "./"):
    """Configure IREE all simple repository dependencies that come from submodules.

    This assumes you have a directory that includes IREE and all its submodules.
    Note that fetching a GitHub archive does not include submodules.
    Yes it is necessary to have both the workspace alias and path argument...

    Args:
      iree_repo_alias: The alias for the IREE repository.
      iree_path: The path to the IREE repository containing submodules
    """

    maybe(
        native.local_repository,
        name = "com_google_absl",
        path = paths.join(iree_path, "third_party/abseil-cpp"),
    )

    maybe(
        native.local_repository,
        name = "com_google_ruy",
        path = paths.join(iree_path, "third_party/ruy"),
    )

    maybe(
        native.local_repository,
        name = "com_google_googletest",
        path = paths.join(iree_path, "third_party/googletest"),
    )

    maybe(
        native.new_local_repository,
        name = "com_github_dvidelabs_flatcc",
        build_file = iree_repo_alias + "//:build_tools/third_party/flatcc/BUILD.overlay",
        path = paths.join(iree_path, "third_party/flatcc"),
    )

    # TODO(scotttodd): TensorFlow is squatting on the vulkan_headers repo name, so
    # we use a temporary one until resolved. Theirs is set to an outdated version.
    maybe(
        native.new_local_repository,
        name = "iree_vulkan_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/vulkan_headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/vulkan_headers"),
    )

    maybe(
        native.new_local_repository,
        name = "vulkan_memory_allocator",
        build_file = iree_repo_alias + "//:build_tools/third_party/vulkan_memory_allocator/BUILD.overlay",
        path = paths.join(iree_path, "third_party/vulkan_memory_allocator"),
    )

    maybe(
        native.local_repository,
        name = "spirv_headers",
        path = paths.join(iree_path, "third_party/spirv_headers"),
    )

    maybe(
        native.local_repository,
        name = "mlir-hlo",
        path = paths.join(iree_path, "third_party/mlir-hlo"),
    )

    maybe(
        native.local_repository,
        name = "com_google_benchmark",
        path = paths.join(iree_path, "third_party/benchmark"),
    )

    maybe(
        native.new_local_repository,
        name = "renderdoc_api",
        build_file = iree_repo_alias + "//:build_tools/third_party/renderdoc_api/BUILD.overlay",
        path = paths.join(iree_path, "third_party/renderdoc_api"),
    )

    maybe(
        native.new_local_repository,
        name = "cpuinfo",
        build_file = iree_repo_alias + "//:build_tools/third_party/cpuinfo/BUILD.overlay",
        path = paths.join(iree_path, "third_party/cpuinfo"),
    )

    maybe(
        native.new_local_repository,
        name = "pffft",
        build_file = iree_repo_alias + "//:build_tools/third_party/pffft/BUILD.overlay",
        path = paths.join(iree_path, "third_party/pffft"),
    )

    maybe(
        native.new_local_repository,
        name = "half",
        build_file = iree_repo_alias + "//:build_tools/third_party/half/BUILD.overlay",
        path = paths.join(iree_path, "third_party/half"),
    )

    maybe(
        native.new_local_repository,
        name = "spirv_cross",
        build_file = iree_repo_alias + "//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = paths.join(iree_path, "third_party/spirv_cross"),
    )
