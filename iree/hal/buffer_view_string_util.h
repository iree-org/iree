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

// Utilities for parsing and printing BufferView, mostly useful for
// testing. The format is as described in
// https://github.com/google/iree/tree/master/iree/base/buffer_string_util.h

#ifndef IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_
#define IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/status.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer_view.h"

namespace iree {
namespace hal {

// Parses a BufferView encoded in a string.
// If an |allocator| is provided the buffer will be allocated as host-local and
// device-visible. Otherwise, buffers will be host-local.
// The format accepted matches that produced by PrintBufferViewToString.
StatusOr<BufferView> ParseBufferViewFromString(
    absl::string_view buffer_view_str, hal::Allocator* allocator = nullptr);

// Prints a BufferView to a string encoded in the canonical format.
StatusOr<std::string> PrintBufferViewToString(const BufferView& buffer_view,
                                              BufferDataPrintMode print_mode,
                                              size_t max_entries);
Status PrintBufferViewToString(const BufferView& buffer_view,
                               BufferDataPrintMode print_mode,
                               size_t max_entries, std::string* out_result);

// Prints a BufferView to a stream encoded in the canonical format.
Status PrintBufferViewToStream(const BufferView& buffer_view,
                               BufferDataPrintMode print_mode,
                               size_t max_entries, std::ostream* stream);

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_
