# TODO(scotttodd): remove after moving out of experimental/, use bazel_to_cmake
set(IREE_PACKAGE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
# Canonicalize path.
cmake_path(ABSOLUTE_PATH IREE_PACKAGE_ROOT_DIR
  BASE_DIRECTORY ${IREE_PACKAGE_ROOT_DIR}
  NORMALIZE
  OUTPUT_VARIABLE IREE_PACKAGE_ROOT_DIR)
set(IREE_PACKAGE_ROOT_PREFIX iree)

iree_add_all_subdirs()

iree_cc_library(
  NAME
    webgpu
  HDRS
    "api.h"
  SRCS
    "api.h"
    "bind_group_cache.c"
    "bind_group_cache.h"
    "buffer.c"
    "buffer.h"
    "builtins.c"
    "builtins.h"
    "command_buffer.c"
    "command_buffer.h"
    "executable.c"
    "executable.h"
    "nop_event.c"
    "nop_event.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "nop_semaphore.c"
    "nop_semaphore.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "simple_allocator.c"
    "simple_allocator.h"
    "staging_buffer.c"
    "staging_buffer.h"
    "webgpu_device.c"
    "webgpu_device.h"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::flatcc::parsing
    iree::base::tracing
    iree::hal
    iree::experimental::webgpu::platform
    iree::experimental::webgpu::shaders
    iree::hal::utils::buffer_transfer
    iree::schemas::wgsl_executable_def_c_fbs
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
