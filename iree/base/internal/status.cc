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

#include "iree/base/internal/status.h"

#include <memory>

namespace iree {

std::ostream& operator<<(std::ostream& os, const StatusCode& x) {
  os << StatusCodeToString(x);
  return os;
}

// static
IREE_MUST_USE_RESULT std::string Status::ToString(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return "OK";
  }
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return "<!>";
  }
  std::string result(buffer_length, '\0');
  if (IREE_UNLIKELY(!iree_status_format(status, result.size() + 1,
                                        const_cast<char*>(result.data()),
                                        &buffer_length))) {
    return "<!>";
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

}  // namespace iree
