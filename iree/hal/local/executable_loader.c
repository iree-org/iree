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

#include "iree/hal/local/executable_loader.h"

void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_loader_t* out_base_loader) {
  iree_atomic_ref_count_init(&out_base_loader->ref_count);
  out_base_loader->vtable = vtable;
}

void iree_hal_executable_loader_retain(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader)) {
    iree_atomic_ref_count_inc(&executable_loader->ref_count);
  }
}

void iree_hal_executable_loader_release(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader) &&
      iree_atomic_ref_count_dec(&executable_loader->ref_count) == 1) {
    executable_loader->vtable->destroy(executable_loader);
  }
}

bool iree_hal_executable_loader_query_support(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_executable_format_t executable_format,
    iree_hal_executable_caching_mode_t caching_mode) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  return executable_loader->vtable->query_support(
      executable_loader, executable_format, caching_mode);
}

iree_status_t iree_hal_executable_loader_try_load(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_executable_layout_t* executable_layout,
    iree_hal_executable_format_t executable_format,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(executable_data.data);
  IREE_ASSERT_ARGUMENT(out_executable);
  return executable_loader->vtable->try_load(
      executable_loader, executable_layout, executable_format, caching_mode,
      executable_data, out_executable);
}
