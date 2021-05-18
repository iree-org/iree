# Copyright 2021 Google LLC

iree_add_all_subdirs()

if(${IREE_BUILD_EXPERIMENTAL_ROCM})

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::base::core_headers
    iree::base::status
    iree::base::tracing
    iree::hal
    experimental::rocm
  INCLUDES
  "${CMAKE_CURRENT_LIST_DIR}/../../.."
  DEFINES
    "IREE_BUILD_EXPERIMENTAL_ROCM=1"
  PUBLIC
)

endif()

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
