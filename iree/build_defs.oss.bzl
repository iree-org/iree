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

# Target to the FileCheck binary.
INTREE_FILECHECK_TARGET = "@llvm-project//llvm:FileCheck"

IREE_CUDA_DEPS = ["//iree/hal/cuda/registration"]

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

def iree_build_test(name, targets):
    """Dummy rule to ensure that targets build.

    This is currently undefined in bazel and is preserved for compatibility.
    """
    pass

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
