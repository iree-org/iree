"""Common Bazel definitions for IREE."""

load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@iree_native_python//:build_defs.bzl", "py_extension")
load("@iree_core//build_tools/third_party/glslang:build_defs.bzl", "glsl_vulkan")
load("@rules_python//python:defs.bzl", "py_library")

def platform_trampoline_deps(basename):
    """Produce a list of deps for the given `basename` platform target.

    Example:
      "file_mapping" -> ["//iree/base/internal/file_mapping_internal"]

    This is used for compatibility with various methods of including the
    library in foreign source control systems.

    Args:
      basename: Library name prefix for a library in base/internal.
    Returns:
      A list of dependencies for depending on the library in a platform
      sensitive way.
    """
    return [
        "//iree/base/internal:%s_internal" % basename,
    ]

# A platform-sensitive list of copts for the Vulkan loader.
PLATFORM_VULKAN_LOADER_COPTS = select({
    "//iree/hal/vulkan:native_vk": [],
    "//iree/hal/vulkan:swiftshader_vk": [],
    "//conditions:default": [],
})

# A platform-sensitive list of dependencies for non-test targets using Vulkan.
PLATFORM_VULKAN_DEPS = select({
    "//iree/hal/vulkan:native_vk": [],
    "//iree/hal/vulkan:swiftshader_vk": [],
    "//conditions:default": [],
})

# A platform-sensitive list of dependencies for tests using Vulkan.
PLATFORM_VULKAN_TEST_DEPS = [
    "@com_google_googletest//:gtest_main",
]

def iree_py_library(**kwargs):
    """Compatibility py_library which has bazel compatible args."""

    # This is used when args are needed that are incompatible with upstream.
    # Presently, this includes:
    #   imports
    py_library(**kwargs)

def iree_py_extension(deps = [], **kwargs):
    """Delegates to the real py_extension."""
    py_extension(
        deps = ["@iree_native_python//:python_headers"] + deps,
        **kwargs
    )

def iree_build_test(name, targets):
    """Dummy rule to ensure that targets build.

    This is currently undefined in bazel and is preserved for compatibility.
    """
    pass

def iree_setup_lit_package(data):
    """Should be called once per test package that contains globbed lit tests.

    Args:
      data: Additional, project specific data deps to add.
    """

    # Bundle together all of the test utilities that are used by tests.
    native.filegroup(
        name = "lit_test_utilities",
        testonly = True,
        data = data + [
            "@llvm//:FileCheck",
        ],
    )

def iree_glob_lit_tests(
        data = [":lit_test_utilities"],
        driver = "//iree/tools:run_lit.sh",
        test_file_exts = ["mlir"]):
    """Globs lit test files into tests for a package.

    For most packages, the defaults suffice. Packages that include this must
    also include a call to iree_setup_lit_package().

    Args:
      data: Data files to include/build.
      driver: Test driver.
      test_file_exts: File extensions to glob.
    """
    for test_file_ext in test_file_exts:
        test_files = native.glob([
            "*.%s" % (test_file_ext,),
            "**/*.%s" % (test_file_ext,),
        ])
        for test_file in test_files:
            test_file_location = "$(location %s)" % (test_file,)
            native.sh_test(
                name = "%s.test" % (test_file,),
                size = "small",
                srcs = [driver],
                data = data + [test_file],
                args = [test_file_location],
            )

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

def iree_glsl_vulkan(**kwargs):
    glsl_vulkan(**kwargs)
