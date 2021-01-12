# Copyright 2019 Google LLC
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

# Workspace file for the IREE project.
# buildozer: disable=positional-args

workspace(name = "iree_core")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

###############################################################################
# Skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()
###############################################################################

###############################################################################
# llvm-project

maybe(
    local_repository,
    name = "llvm_bazel",
    path = "third_party/llvm-bazel/llvm-bazel",
)

load("@llvm_bazel//:zlib.bzl", "llvm_zlib_disable")

maybe(
    llvm_zlib_disable,
    name = "llvm_zlib",
)

load("@llvm_bazel//:terminfo.bzl", "llvm_terminfo_disable")

maybe(
    llvm_terminfo_disable,
    name = "llvm_terminfo",
)

load("@llvm_bazel//:configure.bzl", "llvm_configure")

maybe(
    llvm_configure,
    name = "llvm-project",
    src_path = "third_party/llvm-project",
    src_workspace = "@iree_core//:WORKSPACE",
)
###############################################################################

###############################################################################
# Find and configure the Vulkan SDK, if installed.
load("//build_tools/third_party/vulkan_sdk:repo.bzl", "vulkan_sdk_setup")

maybe(
    vulkan_sdk_setup,
    name = "vulkan_sdk",
)
###############################################################################
# All other IREE submodule dependencies

load("//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps")

configure_iree_submodule_deps()

###############################################################################
# bazel toolchains rules for remote execution (https://releases.bazel.build/bazel-toolchains.html).
http_archive(
    name = "bazel_toolchains",
    sha256 = "8c9728dc1bb3e8356b344088dfd10038984be74e1c8d6e92dbb05f21cabbb8e4",
    strip_prefix = "bazel-toolchains-3.7.1",
    urls = [
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/3.7.1/bazel-toolchains-3.7.1.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/3.7.1/bazel-toolchains-3.7.1.tar.gz",
    ],
)

load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(
    name = "rbe_default",
    base_container_digest = "sha256:1a8ed713f40267bb51fe17de012fa631a20c52df818ccb317aaed2ee068dfc61",
    digest = "sha256:d69c260b98a97ad430d34c4591fb2399e00888750f5d47ede00c1e6f3e774e5a",
    registry = "gcr.io",
    repository = "iree-oss/rbe-toolchain",
    use_checked_in_confs = "Force",
)

###############################################################################
