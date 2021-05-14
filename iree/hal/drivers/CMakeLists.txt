# Copyright 2020 Google LLC
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

# Doesn't use bazel_to_cmake because of custom configuration vars

set(IREE_HAL_DRIVER_MODULES)
if(${IREE_HAL_DRIVER_CUDA})
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::cuda::registration)
endif()
if(${IREE_HAL_DRIVER_DYLIB})
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::dylib::registration)
  # TODO(benvanik): add a IREE_HAL_DRIVER_DYLIB_SYNC or global flag.
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::dylib::registration::sync)
endif()
if(${IREE_HAL_DRIVER_VMLA})
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::vmla::registration)
endif()
if(${IREE_HAL_DRIVER_VMVX})
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::vmvx::registration)
endif()
if(${IREE_HAL_DRIVER_VULKAN})
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::vulkan::registration)
endif()

iree_cc_library(
  NAME
    drivers
  HDRS
    "init.h"
  SRCS
    "init.c"
  DEPS
    iree::base
    iree::base::tracing
    ${IREE_HAL_DRIVER_MODULES}
  PUBLIC
)
