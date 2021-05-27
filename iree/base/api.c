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

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version) {
  if (!out_actual_version) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_api_version_t actual_version = IREE_API_VERSION_0;
  *out_actual_version = actual_version;
  return expected_version == actual_version
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "IREE version mismatch; application expected "
                                "%d but IREE is compiled as %d",
                                expected_version, actual_version);
}
