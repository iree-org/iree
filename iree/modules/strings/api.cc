// Copyright 2020 Google LLC
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

#include <sstream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/strings/api.h"
#include "iree/modules/strings/api_detail.h"
#include "iree/modules/strings/strings_module.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

extern "C" iree_status_t strings_string_create(iree_string_view_t value,
                                               iree_allocator_t allocator,
                                               strings_string_t** out_message) {
  // Note that we allocate the message and the string value together.
  strings_string_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(strings_string_t) + value.size, (void**)&message));
  message->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  message->allocator = allocator;
  message->value.data = ((const char*)message) + sizeof(strings_string_t);
  message->value.size = value.size;
  memcpy((void*)message->value.data, value.data, message->value.size);
  *out_message = message;
  return IREE_STATUS_OK;
}

extern "C" iree_status_t strings_string_tensor_create(
    iree_allocator_t allocator, const iree_string_view_t* value,
    int64_t value_count, const int32_t* shape, size_t rank,
    strings_string_tensor_t** out_message) {
  // TODO(suderman): Use separate allocation for each string. More ref counters
  // but prevents constantly copying.

  // Validate the count is correct.
  size_t count = 1;
  for (int i = 0; i < rank; i++) {
    count *= shape[i];
  }

  if (count != value_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // Compute our total memory requirements
  size_t string_bytes = 0;
  for (int i = 0; i < value_count; i++) {
    string_bytes += value[i].size;
  }

  const size_t shape_bytes = rank * sizeof(int32_t);
  const size_t string_view_bytes = value_count * sizeof(iree_string_view_t);
  const size_t byte_count = sizeof(strings_string_tensor_t) + shape_bytes +
                            string_view_bytes + string_bytes;

  // Allocate and compute byte offsets.
  strings_string_tensor_t* message = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, byte_count, (void**)&message));

  char* shape_ptr = ((char*)message) + sizeof(strings_string_tensor_t);
  char* string_view_ptr = shape_ptr + shape_bytes;
  char* contents_ptr = string_view_ptr + string_view_bytes;

  // Setup the string tensor structure.
  message->shape = (int32_t*)shape_ptr;
  message->values = (iree_string_view_t*)string_view_ptr;
  message->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  message->allocator = allocator;

  // Set string tensor values.
  message->rank = rank;
  message->count = count;

  // Copy the shape.
  memcpy((void*)message->shape, shape, rank * sizeof(int32_t));

  // Copy and allocate each string.
  for (int i = 0; i < count; i++) {
    const auto& src = value[i];
    auto& dest = message->values[i];

    dest.data = (char*)contents_ptr;
    dest.size = src.size;
    memcpy((void*)dest.data, src.data, src.size);
    contents_ptr += src.size;
  }

  *out_message = message;
  return IREE_STATUS_OK;
}

// Returns the count of elements in the tensor.
iree_status_t strings_string_tensor_get_count(
    const strings_string_tensor_t* tensor, size_t* count) {
  if (!tensor) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *count = tensor->count;
  return IREE_STATUS_OK;
}

// returns the list of stored string views.
iree_status_t strings_string_tensor_get_elements(
    const strings_string_tensor_t* tensor, iree_string_view_t* strs,
    size_t count, size_t offset) {
  if (!tensor || offset < 0 || offset + count > tensor->count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < count; i++) {
    strs[i] = tensor->values[i + offset];
  }
  return IREE_STATUS_OK;
}

// Returns the rank of the tensor.
iree_status_t strings_string_tensor_get_rank(
    const strings_string_tensor_t* tensor, int32_t* rank) {
  if (!tensor || !rank) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *rank = tensor->rank;
  return IREE_STATUS_OK;
}

// Returns the shape of the tensor.
iree_status_t strings_string_tensor_get_shape(
    const strings_string_tensor_t* tensor, int32_t* shape, size_t rank) {
  if (!tensor || (!shape && rank != 0) || rank != tensor->rank) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  for (int i = 0; i < rank; i++) {
    shape[i] = tensor->shape[i];
  }
  return IREE_STATUS_OK;
}

// Returns the store string view using the provided indices.
iree_status_t strings_string_tensor_get_element(
    const strings_string_tensor_t* tensor, int32_t* indices, size_t rank,
    iree_string_view_t* str) {
  if (!tensor || !indices || rank != tensor->rank) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  size_t index = 0;
  for (int i = 0; i < rank; i++) {
    if (indices[i] >= tensor->shape[i]) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }

    index = index * tensor->shape[i] + indices[i];
  }

  *str = tensor->values[index];
  return IREE_STATUS_OK;
}

void strings_string_destroy(void* ptr) {
  strings_string_t* message = (strings_string_t*)ptr;
  iree_allocator_free(message->allocator, ptr);
}

void strings_string_tensor_destroy(void* ptr) {
  strings_string_t* message = (strings_string_t*)ptr;
  iree_allocator_free(message->allocator, ptr);
}
