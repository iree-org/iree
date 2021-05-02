// Copyright 2021 Google LLC
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

#include "iree/base/string_view.h"

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"

IREE_API_EXPORT bool iree_string_view_atoi_int32(iree_string_view_t value,
                                                 int32_t* out_value) {
  return absl::SimpleAtoi(absl::string_view(value.data, value.size), out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_uint32(iree_string_view_t value,
                                                  uint32_t* out_value) {
  return absl::SimpleAtoi(absl::string_view(value.data, value.size), out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_int64(iree_string_view_t value,
                                                 int64_t* out_value) {
  return absl::SimpleAtoi(absl::string_view(value.data, value.size), out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_uint64(iree_string_view_t value,
                                                  uint64_t* out_value) {
  return absl::SimpleAtoi(absl::string_view(value.data, value.size), out_value);
}

IREE_API_EXPORT bool iree_string_view_atof(iree_string_view_t value,
                                           float* out_value) {
  return absl::SimpleAtof(absl::string_view(value.data, value.size), out_value);
}

IREE_API_EXPORT bool iree_string_view_atod(iree_string_view_t value,
                                           double* out_value) {
  return absl::SimpleAtod(absl::string_view(value.data, value.size), out_value);
}
