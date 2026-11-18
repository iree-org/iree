# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fetches AMDGPU device bitcode from a pinned official ROCm package."""

def _amdgpu_device_libs_repository_impl(repository_ctx):
    # Share the package pin and archive layout with the CMake compiler build.
    package = json.decode(repository_ctx.read(repository_ctx.attr._package))
    build_file = repository_ctx.path(repository_ctx.attr._build_file)

    # A .deb is an ar archive containing a compressed data tarball. Both layers
    # are extracted by Bazel, without requiring dpkg or a host ROCm install.
    repository_ctx.download_and_extract(
        url = package["url"],
        sha256 = package["sha256"],
        canonical_id = package["url"],
        output = "_package",
        type = "deb",
    )
    repository_ctx.extract(
        archive = "_package/" + package["payload"],
        stripPrefix = package["bitcode_path"],
    )
    repository_ctx.delete("_package")
    for filename in ["ocml.bc", "ockl.bc"]:
        if not repository_ctx.path(filename).exists:
            fail("ROCm package is missing required device bitcode: " + filename)
    repository_ctx.symlink(build_file, "BUILD.bazel")

amdgpu_device_libs_repository = repository_rule(
    implementation = _amdgpu_device_libs_repository_impl,
    attrs = {
        "_package": attr.label(
            default = Label("//:build_tools/third_party/amdgpu_device_libs/rocm_package.json"),
            allow_single_file = True,
        ),
        "_build_file": attr.label(
            default = Label("//:build_tools/third_party/amdgpu_device_libs/BUILD.overlay"),
            allow_single_file = True,
        ),
    },
    doc = "Exposes the official ROCm device libraries as @amdgpu_device_libs//:bitcode.",
)
