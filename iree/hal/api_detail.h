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
//
// Additional definitions for internal users of the api. This should only
// be included from internal implementation files.

#ifndef IREE_HAL_API_DETAIL_H_
#define IREE_HAL_API_DETAIL_H_

#include "iree/hal/api.h"
#include "iree/hal/buffer_view.h"

namespace iree {
namespace hal {

// In the API, buffer views are ref objects, and this allows parts of the
// API outside of the HAL to work with them.
struct iree_hal_buffer_view : public RefObject<iree_hal_buffer_view> {
  BufferView impl;
  iree_allocator_t allocator;

  static void Delete(iree_hal_buffer_view* ptr) {
    ptr->allocator.free(ptr->allocator.self, ptr);
  }
};

}  // namespace hal
}  // namespace iree

#endif
