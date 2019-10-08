"""Common Bazel definitions for IREE."""

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

# A platform-sensitive list of dependencies for tests that use Vulkan.
#
# Define "IREE_VK=swiftshader" to include SwiftShader (if it is available).
PLATFORM_VULKAN_TEST_DEPS = [
    "//testing/base/public:gunit_main",

    # TODO(b/138220713): Support SwiftShader use.
]
