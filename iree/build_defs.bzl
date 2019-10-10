"""Common Bazel definitions for IREE."""

load("//build_tools/third_party/glslang:build_defs.bzl", "glsl_vulkan")
load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

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
#
# These may set IREE_VK_ICD_FILENAMES to a path where a ICD manifest can be found,
# to control which ICDs are available to applications.
PLATFORM_VULKAN_LOADER_COPTS = select({
    "//iree/hal/vulkan:native_vk": [],
    "//iree/hal/vulkan:swiftshader_vk": [
        # TODO(b/138220713): Support SwiftShader use.
    ],
    "//conditions:default": [],
})

# A platform-sensitive list of dependencies for non-test targets using Vulkan.
#
# Define "IREE_VK=swiftshader" to include SwiftShader (if it is available).
PLATFORM_VULKAN_DEPS = select({
    "//iree/hal/vulkan:native_vk": [],
    "//iree/hal/vulkan:swiftshader_vk": [
        # TODO(b/138220713): Support SwiftShader use.
    ],
    "//conditions:default": [],
})

# A platform-sensitive list of dependencies for tests using Vulkan.
#
# Define "IREE_VK=swiftshader" to include SwiftShader (if it is available).
PLATFORM_VULKAN_TEST_DEPS = [
    "@com_google_googletest//:gtest_main",

    # TODO(b/138220713): Support SwiftShader use.
]

def iree_build_test(name, targets):
    """Dummy rule to ensure that targets build.

    This is currently undefined in bazel and is preserved for compatibility.
    """
    pass

def iree_flatbuffer_cc_library(**kwargs):
    """Wrapper for the flatbuffer_cc_library."""

    # TODO(laurenzo): The bazel rule for reflections seems broken in OSS
    # builds. Fix it and enable by default.
    flatbuffer_cc_library(gen_reflections = False, **kwargs)

def iree_cc_embed_data(**kwargs):
    """Wrapper for generating embedded data objects."""

    # TODO(laurenzo): Implement me for OSS builds.
    pass

def iree_glob_lit_tests(**kwargs):
    print("TODO: glob_lit_tests is presently a no-op")

def iree_glsl_vulkan(**kwargs):
    glsl_vulkan(**kwargs)
