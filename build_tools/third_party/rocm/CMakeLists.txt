# Copyright 2021 Google LLC
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
if(NOT ${IREE_BUILD_EXPERIMENTAL_ROCM})
  return()
endif()

if(NOT ROCM_HEADERS_API_ROOT)
  set(ROCM_HEADERS_API_ROOT "/opt/rocm/include")
endif()

if (EXISTS ${ROCM_HEADERS_API_ROOT})
  message(STATUS "ROCm Header Path: ${ROCM_HEADERS_API_ROOT}")
else()
  message(SEND_ERROR "Could not locate ROCm: ${ROCM_HEADERS_API_ROOT}")
endif()

external_cc_library(
  PACKAGE
    rocm_headers
  NAME
    rocm_headers
  ROOT
    ${ROCM_HEADERS_API_ROOT}
  HDRS
    "hip/hip_runtime.h"
  INCLUDES
    ${ROCM_HEADERS_API_ROOT}
)

unset(ROCM_HEADERS_API_ROOT)