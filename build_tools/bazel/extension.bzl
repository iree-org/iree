# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helper functions for configuring IREE and dependent project WORKSPACE files."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "@bazel_tools//tools/build_defs/repo:local.bzl",
    "local_repository",
    "new_local_repository",
)

CUDA_TOOLKIT_ROOT_ENV_KEY = "IREE_CUDA_TOOLKIT_ROOT"

# Our CI docker images use a stripped down CUDA directory tree in some
# images, and it is tailored just to support building key elements.
# When this is done, the IREE_CUDA_DEPS_DIR env var is set, and we
# respect that here in order to match the CMake side (which needs it
# because CUDA toolkit detection differs depending on whether it is
# stripped down or not).
# TODO: Simplify this on the CMake/docker side and update here to match.
# TODO(#15332): Dockerfiles no longer include these deps. Simplify.
CUDA_DEPS_DIR_FOR_CI_ENV_KEY = "IREE_CUDA_DEPS_DIR"

def cuda_auto_configure_impl(repository_ctx):
    env = repository_ctx.os.environ
    cuda_toolkit_root = None

    # TODO(aaronmondal): This is not the way.
    iree_repo_alias = repository_ctx.modules[0].tags.configure[0].iree_repo_alias

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

    build_file_content = repository_ctx.read(
        Label("%s//:build_tools/third_party/cuda/BUILD.template" % iree_repo_alias),
    ).replace(
        "%ENABLED%",
        "True" if cuda_toolkit_root else "False",
    ).replace(
        "%LIBDEVICE_REL_PATH%",
        libdevice_rel_path if cuda_toolkit_root else "BUILD.bazel",
    ).replace(
        "%IREE_REPO_ALIAS%",
        iree_repo_alias,
    )

    new_local_repository(
        name = "iree_cuda",
        build_file_content = build_file_content,
        path = cuda_toolkit_root or "/opt/cuda",
    )

cuda_auto_configure = repository_rule(
    environ = [
        CUDA_DEPS_DIR_FOR_CI_ENV_KEY,
        CUDA_TOOLKIT_ROOT_ENV_KEY,
    ],
    implementation = cuda_auto_configure_impl,
    attrs = {
        "iree_repo_alias": attr.string(default = "@iree_core"),
    },
)

iree_cuda_deps = module_extension(
    doc = "",
    environ = [
        CUDA_DEPS_DIR_FOR_CI_ENV_KEY,
        CUDA_TOOLKIT_ROOT_ENV_KEY,
    ],
    implementation = cuda_auto_configure_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "iree_repo_alias": attr.string(default = "@iree_core"),
            },
        ),
    },
)

# def configure_iree_cuda_deps(iree_repo_alias = None):
#     maybe(
#         cuda_auto_configure,
#         name = "iree_cuda",
#         iree_repo_alias = iree_repo_alias,
#     )

def _iree_submodule_deps_extension_impl(ctx):
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

    iree_repo_alias = ctx.modules[0].tags.configure[0].iree_repo_alias
    iree_path = ctx.modules[0].tags.configure[0].iree_path

    local_repository(
        name = "com_google_googletest",
        path = paths.join(iree_path, "third_party/googletest"),
    )

    new_local_repository(
        name = "com_github_dvidelabs_flatcc",
        build_file = iree_repo_alias + "//:build_tools/third_party/flatcc/BUILD.overlay",
        path = paths.join(iree_path, "third_party/flatcc"),
    )

    new_local_repository(
        name = "vulkan_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/vulkan_headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/vulkan_headers"),
    )

    local_repository(
        name = "stablehlo",
        path = paths.join(iree_path, "third_party/stablehlo"),
    )

    local_repository(
        name = "com_google_benchmark",
        path = paths.join(iree_path, "third_party/benchmark"),
    )

    local_repository(
        name = "cpuinfo",
        path = paths.join(iree_path, "third_party/cpuinfo"),
    )

    new_local_repository(
        name = "spirv_cross",
        build_file = iree_repo_alias + "//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = paths.join(iree_path, "third_party/spirv_cross"),
    )

    new_local_repository(
        name = "tracy_client",
        build_file = iree_repo_alias + "//:build_tools/third_party/tracy_client/BUILD.overlay",
        path = paths.join(iree_path, "third_party/tracy"),
    )

    new_local_repository(
        name = "nccl",
        build_file = iree_repo_alias + "//:build_tools/third_party/nccl/BUILD.overlay",
        path = paths.join(iree_path, "third_party/nccl"),
    )

    new_local_repository(
        name = "hsa_runtime_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/hsa-runtime-headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/hsa-runtime-headers"),
    )

    new_local_repository(
        name = "webgpu_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/webgpu-headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/webgpu-headers"),
    )


iree_submodule_deps = module_extension(
    doc = "",

    implementation = _iree_submodule_deps_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "iree_repo_alias": attr.string(default = "@"),
                "iree_path": attr.string(default = "./"),
            },
        ),
    },
)
