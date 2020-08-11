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

"""Common Bazel definitions for IREE."""

load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", _cc_binary = "cc_binary", _cc_library = "cc_library")

# Target to the FileCheck binary.
INTREE_FILECHECK_TARGET = "@llvm-project//llvm:FileCheck"

def platform_trampoline_deps(basename, path = "base"):
    """Produce a list of deps for the given `basename` platform target.

    Example:
      "file_mapping" -> ["//iree/base/internal/file_mapping_internal"]

    This is used for compatibility with various methods of including the
    library in foreign source control systems.

    Args:
      basename: Library name prefix for a library in iree/[path]/internal.
      path: Folder name to work within.
    Returns:
      A list of dependencies for depending on the library in a platform
      sensitive way.
    """
    return [
        "//iree/%s/internal:%s_internal" % (path, basename),
    ]

# A platform-sensitive list of dependencies for non-test targets using Vulkan.
PLATFORM_VULKAN_DEPS = select({
    "//iree/hal/vulkan:native_vk": [],
    "//iree/hal/vulkan:swiftshader_vk": [],
    "//conditions:default": [],
})

# A platform-sensitive list of dependencies for tests using Vulkan.
PLATFORM_VULKAN_TEST_DEPS = [
    "//iree/testing:gtest_main",
]

# Driver modules that register themselves at link time.
IREE_DRIVER_MODULES = [
    "//iree/hal/dylib:dylib_driver_module",
    "//iree/hal/vmla:vmla_driver_module",
    "//iree/hal/vulkan:vulkan_driver_module",
    "//iree/hal/llvmjit:llvmjit_driver_module",
]

# Aliases to the Starlark cc rules.
cc_library = _cc_library

def iree_build_test(name, targets):
    """Dummy rule to ensure that targets build.

    This is currently undefined in bazel and is preserved for compatibility.
    """
    pass

# The OSS build currently has issues with generating flatbuffer reflections.
# It is hard-coded to disabled here (and in iree_flatbuffer_cc_library) until triaged/fixed.
FLATBUFFER_SUPPORTS_REFLECTIONS = False

def iree_flatbuffer_cc_library(**kwargs):
    """Wrapper for the flatbuffer_cc_library."""

    # TODO(laurenzo): The bazel rule for reflections seems broken in OSS
    # builds. Fix it and enable by default.
    flatbuffer_cc_library(
        gen_reflections = False,
        **kwargs
    )

def cc_binary(linkopts = [], **kwargs):
    """Wrapper around low-level cc_binary that adds flags."""
    _cc_binary(
        linkopts = linkopts + [
            # Just include libraries that should be presumed in 2020.
            "-ldl",
            "-lpthread",
        ],
        **kwargs
    )

def iree_cmake_extra_content(content = "", inline = False):
    """Tool for inserting arbitrary content during Bazel->CMake conversion.

    This does nothing in Bazel, while the contents are inserted as-is in
    converted CMakeLists.txt files.

    Args:
      content: The text to insert into the converted file.
      inline: If true, the content will be inserted inline. Otherwise, it will
        be inserted near the top of the converted file.
    """
    pass
