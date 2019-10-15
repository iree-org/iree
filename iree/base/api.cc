// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/api.h"

#include <cstdlib>
#include <string>

#include "iree/base/api_util.h"
#include "iree/base/file_mapping.h"
#include "iree/base/tracing.h"

namespace iree {

//===----------------------------------------------------------------------===//
// iree Core API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version) {
  iree_api_version_t actual_version = IREE_API_VERSION_0;
  *out_actual_version = actual_version;
  return expected_version == actual_version ? IREE_STATUS_OK
                                            : IREE_STATUS_OUT_OF_RANGE;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_alloc(void* self, iree_host_size_t byte_length, void** out_ptr) {
  IREE_TRACE_SCOPE0("iree_allocator_alloc");

  if (!out_ptr) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_ptr = nullptr;

  if (byte_length <= 0) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  *out_ptr = std::malloc(byte_length);
  if (!*out_ptr) {
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  }

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_free(void* self,
                                                                void* ptr) {
  IREE_TRACE_SCOPE0("iree_allocator_free");
  if (ptr) {
    std::free(ptr);
  }
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::FileMapping
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_open_read(iree_string_view_t path, iree_allocator_t allocator,
                            iree_file_mapping_t** out_file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_open_read");

  if (!out_file_mapping) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_file_mapping = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
      auto file_mapping,
      FileMapping::OpenRead(std::string(path.data, path.size)));

  *out_file_mapping =
      reinterpret_cast<iree_file_mapping_t*>(file_mapping.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_retain(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_retain");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_file_mapping_release(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_release");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_byte_span_t IREE_API_CALL
iree_file_mapping_data(iree_file_mapping_t* file_mapping) {
  IREE_TRACE_SCOPE0("iree_file_mapping_data");
  auto* handle = reinterpret_cast<FileMapping*>(file_mapping);
  CHECK(handle) << "NULL file_mapping handle";
  auto data = handle->data();
  return {const_cast<uint8_t*>(data.data()), data.size()};
}

}  // namespace iree
