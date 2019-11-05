# HAL implementation using Vulkan and (likely) SPIR-V executables.

load("//iree:build_defs.bzl", "PLATFORM_VULKAN_LOADER_COPTS", "PLATFORM_VULKAN_TEST_DEPS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# --define=IREE_VK=native to use the native Vulkan drivers (and real hardware).
config_setting(
    name = "native_vk",
    values = {
        "define": "IREE_VK=native",
    },
)

# --define=IREE_VK=swiftshader to use SwiftShader.
config_setting(
    name = "swiftshader_vk",
    values = {
        "define": "IREE_VK=swiftshader",
    },
)

cc_library(
    name = "debug_reporter",
    srcs = ["debug_reporter.cc"],
    hdrs = ["debug_reporter.h"],
    deps = [
        ":dynamic_symbols",
        ":status_util",
        "//iree/base:status",
        "//iree/base:tracing",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "descriptor_pool_cache",
    srcs = ["descriptor_pool_cache.cc"],
    hdrs = ["descriptor_pool_cache.h"],
    deps = [
        ":dynamic_symbols",
        ":handle_util",
        ":status_util",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "descriptor_set_arena",
    srcs = ["descriptor_set_arena.cc"],
    hdrs = ["descriptor_set_arena.h"],
    deps = [
        ":descriptor_pool_cache",
        ":pipeline_executable",
        ":status_util",
        ":vma_allocator",
        "//iree/base:arena",
        "//iree/base:math",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
    ],
)

cc_library(
    name = "direct_command_buffer",
    srcs = ["direct_command_buffer.cc"],
    hdrs = ["direct_command_buffer.h"],
    deps = [
        ":descriptor_pool_cache",
        ":descriptor_set_arena",
        ":dynamic_symbols",
        ":handle_util",
        ":native_event",
        ":pipeline_executable",
        ":status_util",
        ":vma_allocator",
        "//iree/base:math",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "direct_command_queue",
    srcs = ["direct_command_queue.cc"],
    hdrs = ["direct_command_queue.h"],
    deps = [
        ":direct_command_buffer",
        ":dynamic_symbols",
        ":handle_util",
        ":legacy_fence",
        ":native_binary_semaphore",
        ":status_util",
        "//iree/base:arena",
        "//iree/base:memory",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "dynamic_symbols",
    srcs = ["dynamic_symbols.cc"],
    hdrs = [
        "dynamic_symbol_tables.h",
        "dynamic_symbols.h",
    ],
    copts = PLATFORM_VULKAN_LOADER_COPTS,
    linkopts = [
        "-ldl",
    ],
    deps = [
        "//iree/base:file_path",
        "//iree/base:platform_headers",
        "//iree/base:ref_ptr",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:tracing",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    deps = [
        ":status_util",
        ":dynamic_symbols",
        "//iree/base:status_matchers",
    ] + PLATFORM_VULKAN_TEST_DEPS,
)

cc_library(
    name = "extensibility_util",
    srcs = ["extensibility_util.cc"],
    hdrs = ["extensibility_util.h"],
    deps = [
        ":dynamic_symbols",
        ":status_util",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/types:span",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "handle_util",
    hdrs = ["handle_util.h"],
    deps = [
        ":dynamic_symbols",
        ":extensibility_util",
        "//iree/base:ref_ptr",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/utility",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "legacy_fence",
    srcs = ["legacy_fence.cc"],
    hdrs = ["legacy_fence.h"],
    deps = [
        ":handle_util",
        ":status_util",
        "//iree/base:intrusive_list",
        "//iree/base:ref_ptr",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:fence",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "native_binary_semaphore",
    srcs = ["native_binary_semaphore.cc"],
    hdrs = ["native_binary_semaphore.h"],
    deps = [
        ":handle_util",
        "//iree/hal:semaphore",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "native_event",
    srcs = ["native_event.cc"],
    hdrs = ["native_event.h"],
    deps = [
        ":handle_util",
        "//iree/hal:event",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "pipeline_cache",
    srcs = ["pipeline_cache.cc"],
    hdrs = ["pipeline_cache.h"],
    deps = [
        ":handle_util",
        ":pipeline_executable",
        ":status_util",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_format",
        "//iree/schemas:spirv_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "pipeline_executable",
    srcs = ["pipeline_executable.cc"],
    hdrs = ["pipeline_executable.h"],
    deps = [
        ":handle_util",
        ":status_util",
        "//iree/base:memory",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_spec",
        "//iree/schemas:spirv_executable_def_cc_fbs",
        "@com_google_absl//absl/container:inlined_vector",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "status_util",
    srcs = ["status_util.cc"],
    hdrs = ["status_util.h"],
    deps = [
        "//iree/base:status",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "vma_allocator",
    srcs = [
        "internal_vk_mem_alloc.cc",
        "internal_vk_mem_alloc.h",
        "vma_allocator.cc",
        "vma_buffer.cc",
    ],
    hdrs = [
        "vma_allocator.h",
        "vma_buffer.h",
    ],
    copts = [
        # Only needed in the implementation cc and not by external users.
        "-DVMA_STATIC_VULKAN_FUNCTIONS=0",
    ],
    deps = [
        ":dynamic_symbols",
        ":handle_util",
        ":status_util",
        "//iree/base:logging",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:buffer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
        "@vulkan_memory_allocator//:impl_header_only",
    ],
)

cc_library(
    name = "vulkan_device",
    srcs = ["vulkan_device.cc"],
    hdrs = ["vulkan_device.h"],
    deps = [
        ":descriptor_pool_cache",
        ":direct_command_buffer",
        ":direct_command_queue",
        ":dynamic_symbols",
        ":extensibility_util",
        ":handle_util",
        ":legacy_fence",
        ":native_binary_semaphore",
        ":native_event",
        ":pipeline_cache",
        ":status_util",
        ":vma_allocator",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:command_buffer_validation",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal:driver",
        "//iree/hal:fence",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "vulkan_driver",
    srcs = ["vulkan_driver.cc"],
    hdrs = ["vulkan_driver.h"],
    deps = [
        ":debug_reporter",
        ":dynamic_symbols",
        ":extensibility_util",
        ":status_util",
        ":vulkan_device",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:device_info",
        "//iree/hal:driver",
        "@com_google_absl//absl/container:inlined_vector",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "vulkan_driver_module",
    srcs = ["vulkan_driver_module.cc"],
    deps = [
        ":dynamic_symbols",
        ":vulkan_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:driver_registry",
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = 1,
)
