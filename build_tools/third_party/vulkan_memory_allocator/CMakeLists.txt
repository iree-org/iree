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

set(VMA_SRC_ROOT
  "${IREE_ROOT_DIR}/third_party/vulkan_memory_allocator/src/"
)

set(VMA_COPTS_BASE
  "-I${VMA_SRC_ROOT}"
)

external_cc_library(
  PACKAGE
    vulkan_memory_allocator
  NAME
    no_prototypes
  ROOT
    ${VMA_SRC_ROOT}
  SRCS
    "vk_mem_alloc.h"
  HDRS
    "vk_mem_alloc.h"
  COPTS
    "-I${VMA_SRC_ROOT}"
    # Only needed in the implementation cc and not by external users.
    "-DVMA_STATIC_VULKAN_FUNCTIONS=0"
  DEPS
    absl::flat_hash_map
    absl::synchronization
    iree::base::logging
  PUBLIC
)
