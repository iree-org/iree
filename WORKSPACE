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
load(":repo_utils.bzl", "maybe")

###############################################################################
# Bazel rules.
http_archive(
    name = "rules_cc",
    sha256 = "cf3b76a90c86c0554c5b10f4b160f05af71d252026b71362c4674e2fb9936cf9",
    strip_prefix = "rules_cc-01d4a48911d5e7591ecb1c06d3b8af47fe872371",
    urls = [
	"https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_cc/archive/01d4a48911d5e7591ecb1c06d3b8af47fe872371.zip",
	"https://github.com/bazelbuild/rules_cc/archive/01d4a48911d5e7591ecb1c06d3b8af47fe872371.zip",
    ],
)

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

###############################################################################

###############################################################################
# bazel toolchains rules for remote execution (https://releases.bazel.build/bazel-toolchains.html).
http_archive(
    name = "bazel_toolchains",
    sha256 = "4d348abfaddbcee0c077fc51bb1177065c3663191588ab3d958f027cbfe1818b",
    strip_prefix = "bazel-toolchains-2.1.0",
    urls = [
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/2.1.0/bazel-toolchains-2.1.0.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/2.1.0.tar.gz",
    ],
)

# Disable RBE until compatibility issues with the experimental_repo_remote_exec
# flag are fixed.
# load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")
# rbe_autoconfig(name = "rbe_default")

###############################################################################

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
# llvm-project
load("@iree_core//build_tools/bazel/third_party_import/llvm-project:configure.bzl", "llvm_configure")
maybe(llvm_configure,
    name = "llvm-project",
    workspace = "@iree_core//:WORKSPACE",
    path = "third_party/llvm-project",
)
###############################################################################

###############################################################################
# Bootstrap TensorFlow.
# Note that we ultimately would like to avoid doing this at the top level like
# this but need to unbundle some of the deps from the tensorflow repo first.
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
###############################################################################

###############################################################################
# Autoconfigure native build repo for python.
load("//bindings/python/build_tools/python:configure.bzl", "python_configure")

# TODO(laurenzo): Scoping to "iree" to avoid conflicts with other things that
# take an opinion until we can isolate.
maybe(python_configure,
    name = "iree_native_python",
)
###############################################################################

###############################################################################
# Find and configure the Vulkan SDK, if installed.
load("//build_tools/third_party/vulkan_sdk:repo.bzl", "vulkan_sdk_setup")
maybe(vulkan_sdk_setup,
    name = "vulkan_sdk",
)
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

# TODO(scotttodd): TensorFlow is squatting on the vulkan_headers repo name, so
# we use a temporary one until resolved. Theirs is set to an outdated version.
maybe(new_local_repository,
    name = "iree_vulkan_headers",
    path = "third_party/vulkan_headers",
    build_file = "build_tools/third_party/vulkan_headers/BUILD.overlay",
)

maybe(new_local_repository,
    name = "vulkan_extensionlayer",
    path = "third_party/vulkan_extensionlayer",
    build_file = "build_tools/third_party/vulkan_extensionlayer/BUILD.overlay",
)

maybe(new_local_repository,
    name = "vulkan_memory_allocator",
    path = "third_party/vulkan_memory_allocator",
    build_file = "build_tools/third_party/vulkan_memory_allocator/BUILD.overlay",
)

maybe(new_local_repository,
    name = "glslang",
    path = "third_party/glslang",
    build_file = "build_tools/third_party/glslang/BUILD.overlay",
)

maybe(local_repository,
    name = "spirv_tools",
    path = "third_party/spirv_tools",
)

maybe(local_repository,
    name = "spirv_headers",
    path = "third_party/spirv_headers",
)

# TODO(laurenzo): TensorFlow is squatting on the pybind11 repo name, so
# we use a temporary one until resolved. Theirs pulls in a bunch of stuff.
maybe(new_local_repository,
    name = "iree_pybind11",
    path = "third_party/pybind11",
    build_file = "build_tools/third_party/pybind11/BUILD.overlay",
)

maybe(local_repository,
    name = "com_google_benchmark",
    path = "third_party/benchmark")

maybe(new_local_repository,
    name = "sdl2",
    path = "third_party/sdl2",
    build_file = "build_tools/third_party/sdl2/BUILD.overlay",
)

maybe(new_local_repository,
    name = "sdl2_config",
    path = "build_tools/third_party/sdl2",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(name = "headers", srcs = glob(["*.h"]))
""",
)

maybe(new_local_repository,
    name = "dear_imgui",
    path = "third_party/dear_imgui",
    build_file = "build_tools/third_party/dear_imgui/BUILD.overlay",
)

maybe(new_local_repository,
    name = "renderdoc_api",
    path = "third_party/renderdoc_api",
    build_file = "build_tools/third_party/renderdoc_api/BUILD.overlay",
)

# Bootstrap TensorFlow deps last so that ours can take precedence.
tf_repositories()
