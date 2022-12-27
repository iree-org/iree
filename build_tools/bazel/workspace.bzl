# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helper functions for configuring IREE and dependent project WORKSPACE files."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_skylib//lib:paths.bzl", "paths")

CUDA_TOOLKIT_ROOT_ENV_KEY = "IREE_CUDA_TOOLKIT_ROOT"

# Our CI docker images use a stripped down CUDA directory tree in some
# images, and it is tailored just to support building key elements.
# When this is done, the IREE_CUDA_DEPS_DIR env var is set, and we
# respect that here in order to match the CMake side (which needs it
# because CUDA toolkit detection differs depending on whether it is
# stripped down or not).
# TODO: Simplify this on the CMake/docker side and update here to match.
CUDA_DEPS_DIR_FOR_CI_ENV_KEY = "IREE_CUDA_DEPS_DIR"

def cuda_auto_configure_impl(repository_ctx):
    env = repository_ctx.os.environ
    cuda_toolkit_root = None

    # Probe environment for CUDA toolkit location.
    env_cuda_toolkit_root = env.get(CUDA_TOOLKIT_ROOT_ENV_KEY)
    env_cuda_deps_dir_for_ci = env.get(CUDA_DEPS_DIR_FOR_CI_ENV_KEY)
    if env_cuda_toolkit_root:
        cuda_toolkit_root = env_cuda_toolkit_root
    elif env_cuda_deps_dir_for_ci:
        cuda_toolkit_root = env_cuda_deps_dir_for_ci

    # Symlink the tree.
    libdevice_rel_path = "iree_local/libdevice.bc"
    if cuda_toolkit_root != None:
        # Symlink top-level directories we care about.
        repository_ctx.symlink(cuda_toolkit_root + "/include", "include")

        # TODO: Should be probing for the libdevice, as it can change from
        # version to version.
        repository_ctx.symlink(
            cuda_toolkit_root + "/nvvm/libdevice/libdevice.10.bc",
            libdevice_rel_path,
        )

    repository_ctx.template(
        "BUILD",
        Label("@iree_core//:build_tools/third_party/cuda/BUILD.template"),
        {
            "%ENABLED%": "True" if cuda_toolkit_root else "False",
            "%LIBDEVICE_REL_PATH%": libdevice_rel_path,
        },
    )

cuda_auto_configure = repository_rule(
    environ = [
        CUDA_DEPS_DIR_FOR_CI_ENV_KEY,
        CUDA_TOOLKIT_ROOT_ENV_KEY,
    ],
    implementation = cuda_auto_configure_impl,
)

def configure_iree_cuda_deps():
    maybe(
        cuda_auto_configure,
        name = "iree_cuda",
    )

def configure_iree_submodule_deps(iree_repo_alias = "@", iree_path = "./"):
    """Configure all of IREE's simple repository dependencies that come from submodules.

    Simple is defined here as just calls to `local_repository` or
    `new_local_repository`. This assumes you have a directory that includes IREE
    and all its submodules. Note that fetching a GitHub archive does not include
    submodules.
    Yes it is necessary to have both the workspace alias and path argument...

    Args:
      iree_repo_alias: The alias for the IREE repository.
      iree_path: The path to the IREE repository containing submodules
    """

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

    maybe(
        native.new_local_repository,
        name = "com_github_yaml_libyaml",
        build_file = iree_repo_alias + "//:build_tools/third_party/libyaml/BUILD.overlay",
        path = paths.join(iree_path, "third_party/libyaml"),
    )

    maybe(
        native.new_local_repository,
        name = "vulkan_headers",
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
        name = "cpuinfo",
        build_file = iree_repo_alias + "//:build_tools/third_party/cpuinfo/BUILD.overlay",
        path = paths.join(iree_path, "third_party/cpuinfo"),
    )

    maybe(
        native.new_local_repository,
        name = "spirv_cross",
        build_file = iree_repo_alias + "//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = paths.join(iree_path, "third_party/spirv_cross"),
    )

    maybe(
        native.new_local_repository,
        name = "torch-mlir-dialects",
        build_file = iree_repo_alias + "//:build_tools/third_party/torch-mlir-dialects/BUILD.overlay",
        path = paths.join(iree_path, "third_party/torch-mlir-dialects"),
    )
